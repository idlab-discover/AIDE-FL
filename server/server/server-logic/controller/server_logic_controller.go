package controller

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service"
	"github.com/fasthttp/websocket"
	"github.com/google/uuid"
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
	prepId                   uuid.UUID
	prepIdMutex              *sync.Mutex
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
		prepIdMutex:              &sync.Mutex{},
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

func (controller *ServerLogicController) getPrepId() uuid.UUID {
	controller.prepIdMutex.Lock()
	uuid := controller.prepId
	controller.prepIdMutex.Unlock()
	return uuid
}

func (controller *ServerLogicController) setNewPrepId() uuid.UUID {
	controller.prepIdMutex.Lock()
	controller.prepId = uuid.New()
	uuid := controller.prepId
	controller.prepIdMutex.Unlock()
	return uuid
}

func (controller *ServerLogicController) Route() {
	http.HandleFunc("/ws", controller.handleWS)
	http.HandleFunc("/start", controller.start)
	http.HandleFunc("/status", controller.status)
	http.HandleFunc("/clients", controller.connectedClients)
}

func (controller *ServerLogicController) handleWS(w http.ResponseWriter, r *http.Request) {
	// Get client ID from HTTP header (no need to check for duplicates since client ID = node name)
	clientId := r.Header.Get("X-Client-ID")

	conn, err := controller.upgrader.Upgrade(w, r, nil)
	exception.PanicLogging(err, "Unable to upgrade connection to WebSocket")

	// Get client address
	clientAddress := conn.RemoteAddr().String()
	slog.Info("Client connected:", "ID", clientId, "address", clientAddress)

	// Add new client to clients
	client := &model.Client{Conn: conn, Id: clientId, Address: clientAddress}
	controller.clientsMutex.Lock()
	controller.clients[clientId] = client
	controller.clientsMutex.Unlock()
	controller.totalClients++

	defer func() {
		controller.clientsMutex.Lock()
		delete(controller.clients, clientId)
		controller.clientsMutex.Unlock()
		controller.totalClients--
		conn.Close()
		slog.Info("Client disconnected:", "ID", clientId, "address", clientAddress)
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
		if generalMessage.Type == "INFO" && generalMessage.Origin == "orchestrator" && strings.HasPrefix(generalMessage.Message, "DONE:") {
			if strings.Split(generalMessage.Message, ":")[1] != controller.getPrepId().String() {
				slog.Info("Reported DONE with mismatching prep ID by", "client", clientId)
				continue
			}

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

		var flowerOut model.FlowerResponse
		json.Unmarshal([]byte(flowerStr), &flowerOut)

		response := model.GeneralResponse{
			Message: "Aborting training due to already ongoing training run",
			Data:    flowerOut.Runs[0],
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

	// Check whether a dataset has been provided for every connected client if there is a datasets entry
	if len(scenario.Datasets) != 0 {
		controller.clientsMutex.Lock()
		for _, client := range controller.clients {
			_, ok := scenario.Datasets[client.Id]
			if !ok {
				slog.Error("No dataset has been provided for", "client", client.Id)

				response := model.GeneralResponse{
					Message: fmt.Sprintf("No dataset has been provided for %s", client.Id),
					Data:    nil,
				}
				jsonData, _ := json.Marshal(response)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				w.Write(jsonData)

				controller.clientsMutex.Unlock()
				return
			}
		}
		controller.clientsMutex.Unlock()
	}

	// Start of training
	controller.setIsTraining(true)

	// Reset error message and done count
	controller.errorMessage = nil
	controller.doneCount = 0

	// Set new preparation ID (is sent to client together with scenario so that DONE messages can be identified to be matching with the current training run request; outdated IDs can be received if one of multiple clients sent an error aborting the run, a new run is starting, and other clients are reporting DONE from the previous run)
	controller.setNewPrepId()

	// Set config for Flower serverapp
	controller.ServerLogicService.SetFlowerConfig(scenario)

	// Send scenario to clients with uuid
	slog.Info("start", "totalClients", controller.totalClients)
	controller.clientsMutex.Lock()
	controller.ServerLogicService.StartClients(scenario, controller.getPrepId(), controller.clients)
	controller.clientsMutex.Unlock()

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

		controller.setIsTraining(false)
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

func (controller *ServerLogicController) connectedClients(w http.ResponseWriter, r *http.Request) {
	slog.Info("Received /clients request")

	controller.clientsMutex.Lock()

	clientList := make([]*model.Client, 0, 2)
	for key := range controller.clients {
		controller.clients[key].Conn = nil // Don't try to return Conn
		clientList = append(clientList, controller.clients[key])
	}

	// Response
	response := model.GeneralResponse{
		Message: "Following clients are connected to the server",
		Data:    clientList,
	}
	jsonData, _ := json.Marshal(response)
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)

	controller.clientsMutex.Unlock()
}
