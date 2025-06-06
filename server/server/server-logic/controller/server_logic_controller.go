package controller

import (
	"encoding/json"
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
	upgrader     websocket.Upgrader
	clients      map[string]*model.Client
	clientsMutex *sync.Mutex
	doneCount    int
	totalClients int
	doneCond     *sync.Cond
}

func NewServerLogicController(serverLogicService *service.ServerLogicService, minioService *service.MinioService, config configuration.Config) *ServerLogicController {
	this := &ServerLogicController{
		MinioService:       *minioService,
		ServerLogicService: *serverLogicService,
		Config:             config,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
		clients:      make(map[string]*model.Client),
		clientsMutex: &sync.Mutex{},
		doneCount:    0,
		totalClients: 0,
		doneCond:     sync.NewCond(&sync.Mutex{}),
	}

	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_BUCKET"))

	return this
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

	// Wait for 'done' message
	for {
		_, msg, err := conn.ReadMessage()
		exception.PanicLogging(err, "Unable to read message from", "client", clientId)

		if string(msg) == "done" {
			controller.doneCond.L.Lock()
			slog.Info("Reported done by", "client", clientId)
			controller.doneCount++
			slog.Info("Progress:", "done", controller.doneCount, "total", controller.totalClients)
			if controller.doneCount >= controller.totalClients {
				controller.doneCond.Broadcast()
			}
		}
		controller.doneCond.L.Unlock()
	}
}

func (controller *ServerLogicController) start(w http.ResponseWriter, r *http.Request) {
	slog.Info("Received /start request")
	slog.Info("start", "totalClients", controller.totalClients)
	controller.clientsMutex.Lock()
	defer controller.clientsMutex.Unlock()

	// Reset done count
	controller.doneCount = 0

	// Get scenario from request
	var scenario model.Scenario
	dec := json.NewDecoder(r.Body)
	err := dec.Decode(&scenario)
	exception.PanicLogging(err, "Unable to decode JSON to scenario")

	// Set config for Flower serverapp
	controller.ServerLogicService.SetFlowerConfig(scenario)

	// Send request to clients
	controller.ServerLogicService.StartClients(scenario, controller.clients)

	// Start training when all clients are done preparing
	controller.doneCond.L.Lock()
	for controller.doneCount < controller.totalClients {
		controller.doneCond.Wait()
	}
	controller.doneCond.L.Unlock()
	slog.Info("All clients reported done")
	jsonStr := controller.ServerLogicService.StartTraining()

	// Response
	var jsonFlower map[string]interface{}
	json.Unmarshal([]byte(jsonStr), &jsonFlower)
	response := model.GeneralResponse{
		Message: "Clients are done preparing, training has started.",
		Data:    jsonFlower,
	}
	jsonData, _ := json.Marshal(response)
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)

	// Watch for creation of safetensors file in shared folder and upload to datastore
	go controller.ServerLogicService.WatchForSafetensors(controller.MinioService.UploadSafetensors)
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
