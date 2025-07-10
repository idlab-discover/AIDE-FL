package controller

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service"
	"github.com/fasthttp/websocket"
)

type ServerLogicController struct {
	service.ServerLogicService
	service.MinioService
	configuration.Config
	upgrader                 websocket.Upgrader
	clients                  map[string]*model.Client
	clientsMutex             *sync.Mutex
	totalClients             int
	doneCount                int
	isTraining               bool
	isTrainingMutex          *sync.Mutex
	latestTrainingRunId      string
	latestTrainingRunIdMutex *sync.Mutex
	doneCond                 *sync.Cond
	errorMessage             *model.GeneralMessage
}

func NewServerLogicController(serverLogicService *service.ServerLogicService, minioService *service.MinioService, config configuration.Config) *ServerLogicController {
	this := &ServerLogicController{
		MinioService:       *minioService,
		ServerLogicService: *serverLogicService,
		Config:             config,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
		clients:                  make(map[string]*model.Client),
		clientsMutex:             &sync.Mutex{},
		totalClients:             0,
		doneCount:                0,
		isTraining:               false,
		isTrainingMutex:          &sync.Mutex{},
		latestTrainingRunIdMutex: &sync.Mutex{},
		doneCond:                 sync.NewCond(&sync.Mutex{}),
	}

	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_BUCKET"))

	return this
}

func (controller *ServerLogicController) getIsTraining() bool {
	controller.isTrainingMutex.Lock()
	isTraining := controller.isTraining
	controller.isTrainingMutex.Unlock()
	return isTraining
}

func (controller *ServerLogicController) setIsTraining(isTraining bool) {
	controller.isTrainingMutex.Lock()
	controller.isTraining = isTraining
	controller.isTrainingMutex.Unlock()
}

func (controller *ServerLogicController) getLatestTrainingRunId() string {
	controller.latestTrainingRunIdMutex.Lock()
	latestTrainingRunId := controller.latestTrainingRunId
	controller.latestTrainingRunIdMutex.Unlock()
	return latestTrainingRunId
}

func (controller *ServerLogicController) setLatestTrainingRunId(latestTrainingRunId string) {
	controller.latestTrainingRunIdMutex.Lock()
	controller.latestTrainingRunId = latestTrainingRunId
	controller.latestTrainingRunIdMutex.Unlock()
}

func (controller *ServerLogicController) Route() {
	http.HandleFunc("/ws", controller.handleWS)
	http.HandleFunc("/start", controller.start)
	http.HandleFunc("/status", controller.status)
}

func (controller *ServerLogicController) handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := controller.upgrader.Upgrade(w, r, nil)
	exception.PanicLogging(err, "Unable to upgrade connection to WebSocket")

	// Use client address as client ID
	clientId := conn.RemoteAddr().String()
	slog.Info("Client connected:", "ip", clientId)

	// Add new client to clients
	client := &model.Client{Conn: conn, Id: clientId}
	controller.clientsMutex.Lock()
	controller.clients[clientId] = client
	controller.totalClients++
	controller.clientsMutex.Unlock()

	defer func() {
		controller.clientsMutex.Lock()
		delete(controller.clients, clientId)
		controller.totalClients--
		controller.clientsMutex.Unlock()
		conn.Close()
		slog.Info("Client disconnected:", "client", clientId)
	}()

	controller.listenForGeneralMessages(conn, clientId)
}

// Given a client connection, listen for incoming general messages
func (controller *ServerLogicController) listenForGeneralMessages(conn *websocket.Conn, clientId string) {
	for {
		_, msg, err := conn.ReadMessage()
		exception.PanicLogging(err, "Unable to read message from", "client", clientId)

		var generalMessage model.GeneralMessage
		err = json.Unmarshal(msg, &generalMessage)
		if err != nil {
			slog.Error("Unable to unmarshal JSON to general message", "error", err)
			continue // Skip bad message
		}

		// Handle DONE message
		if generalMessage.Type == "INFO" && generalMessage.Origin == "orchestrator" && generalMessage.Message == "DONE" {
			controller.doneCond.L.Lock()
			slog.Info("Reported DONE by", "client", clientId)
			controller.doneCount++
			slog.Info("Progress:", "done", controller.doneCount, "total", controller.totalClients)
			if controller.doneCount >= controller.totalClients {
				controller.doneCond.Broadcast()
			}
			controller.doneCond.L.Unlock()
		}

		// Handle ERROR message
		if generalMessage.Type == "ERROR" {
			controller.doneCond.L.Lock()
			slog.Error("ERROR received from", "clientId", clientId)

			controller.errorMessage = &generalMessage

			controller.doneCond.Broadcast()
			controller.doneCond.L.Unlock()
		}
	}
}

func (controller *ServerLogicController) start(w http.ResponseWriter, r *http.Request) {
	slog.Info("Received /start request")

	// Abort if there is an ongoing training run / client preparations
	if controller.getIsTraining() {
		slog.Error("Aborting training due to already ongoing training run")

		// Get Flower details about current run to include in response
		runId := controller.getLatestTrainingRunId()
		flowerStr := controller.ServerLogicService.GetTrainingStatus(runId)

		var flowerOut map[string]interface{}
		json.Unmarshal([]byte(flowerStr), &flowerOut)

		response := model.GeneralResponse{
			Message: "Aborting training due to already ongoing training run",
			Data:    flowerOut["runs"].([]interface{})[0],
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "60")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write(jsonData)

		return
	}

	// Decode body
	scenario, err := controller.ServerLogicService.DecodeScenario(r.Body)
	if err != nil {
		slog.Error("Aborting training due to invalid request body", "error", err)

		response := model.GeneralResponse{
			Message: "Aborting training due to invalid request body",
			Data:    err.Error(),
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write(jsonData)
		return
	}

	// Validate scenario from request
	err = controller.ServerLogicService.ValidateScenario(scenario)
	if err != nil {
		slog.Error("Aborting training due to invalid scenario", "error", err)

		response := model.GeneralResponse{
			Message: "Aborting training due to invalid scenario",
			Data:    err.Error(),
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write(jsonData)
		return
	}

	// Check for ample client connections
	if controller.totalClients < scenario.Strategy.MinAvailableClients {
		slog.Error("Aborting training due to insufficient client connections", "current", controller.totalClients, "required", scenario.Strategy.MinAvailableClients)

		response := model.GeneralResponse{
			Message: fmt.Sprintf("Aborting training due to insufficient client connections (%d out of %d)", controller.totalClients, scenario.Strategy.MinAvailableClients),
			Data:    nil,
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooEarly)
		w.Write(jsonData)
		return
	}

	// Start of training
	controller.setIsTraining(true)

	// Reset error message and done count
	controller.errorMessage = nil
	controller.doneCount = 0

	// Set config for Flower serverapp
	controller.ServerLogicService.SetFlowerConfig(scenario)

	// Send scenario to clients
	slog.Info("start", "totalClients", controller.totalClients)
	controller.clientsMutex.Lock()
	defer controller.clientsMutex.Unlock()
	controller.ServerLogicService.StartClients(scenario, controller.clients)

	// Wait for doneCount to Broadcast (when all clients report DONE or when an ERROR message is received)
	controller.doneCond.L.Lock()
	for controller.doneCount < controller.totalClients && controller.errorMessage == nil {
		controller.doneCond.Wait()
	}
	controller.doneCond.L.Unlock()

	// Abort on error
	if controller.errorMessage != nil {
		slog.Error("Aborting training due to error", "origin", controller.errorMessage.Origin, "message", controller.errorMessage.Message)

		response := model.GeneralResponse{
			Message: "Training run aborted due to client error",
			Data:    controller.errorMessage,
		}
		jsonData, _ := json.Marshal(response)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write(jsonData)
		return
	}

	// Start Flower training if no error message
	slog.Info("All clients reported done")
	jsonStr := controller.ServerLogicService.StartTraining()

	// Response
	var jsonFlower map[string]interface{}
	json.Unmarshal([]byte(jsonStr), &jsonFlower)

	runId, _ := jsonFlower["run-id"].(string)
	controller.setLatestTrainingRunId(runId)

	response := model.GeneralResponse{
		Message: "Clients are done preparing, training has started.",
		Data:    jsonFlower,
	}
	jsonData, _ := json.Marshal(response)
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)

	// Go routine so that HTTP response may already be sent
	go func() {
		// Watch for creation of safetensors file in shared folder and upload to datastore
		controller.ServerLogicService.WatchForSafetensors(controller.MinioService.UploadSafetensors)

		// Free to start another training run
		controller.setIsTraining(false)
	}()
}

func (controller *ServerLogicController) status(w http.ResponseWriter, r *http.Request) {
	slog.Info("Received /status request with", "params", r.URL.Query())

	runId := r.URL.Query().Get("runId")
	jsonStr := controller.ServerLogicService.GetTrainingStatus(runId)

	// Response
	var flowerResponse model.FlowerResponse
	err := json.Unmarshal([]byte(jsonStr), &flowerResponse)
	exception.PanicLogging(err, "Unable to unmarshal json to FlowerResponse")

	response := model.GeneralResponse{
		Message: "Flower results",
		Data:    flowerResponse,
	}
	jsonData, _ := json.Marshal(response)
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)
}
