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
	activeSession            *trainingSession
	activeSessionMutex       *sync.Mutex
	latestTrainingRunId      string
	latestTrainingRunIdMutex *sync.Mutex
}

type trainingSession struct {
	mu                sync.Mutex
	doneCond          *sync.Cond
	prepID            uuid.UUID
	trainingClientIDs map[string]struct{}
	completedClients  map[string]struct{}
	errorMessage      *model.GeneralMessage
}

func newTrainingSession() *trainingSession {
	session := &trainingSession{}
	session.doneCond = sync.NewCond(&session.mu)
	return session
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
		activeSessionMutex:       &sync.Mutex{},
		latestTrainingRunIdMutex: &sync.Mutex{},
	}

	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_MODELS_BUCKET"))
	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_DATASETS_BUCKET"))
	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_CHECKPOINTS_BUCKET"))
	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_METRICS_BUCKET"))

	return this
}

func (controller *ServerLogicController) reserveTrainingSession() (*trainingSession, bool) {
	controller.activeSessionMutex.Lock()
	defer controller.activeSessionMutex.Unlock()

	if controller.activeSession != nil {
		return nil, false
	}

	session := newTrainingSession()
	controller.activeSession = session
	return session, true
}

func (controller *ServerLogicController) currentTrainingSession() *trainingSession {
	controller.activeSessionMutex.Lock()
	defer controller.activeSessionMutex.Unlock()

	return controller.activeSession
}

func (controller *ServerLogicController) finishTrainingSession(session *trainingSession) {
	controller.activeSessionMutex.Lock()
	defer controller.activeSessionMutex.Unlock()

	if controller.activeSession == session {
		controller.activeSession = nil
	}
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

func (controller *ServerLogicController) connectedClientCount() int {
	controller.clientsMutex.Lock()
	defer controller.clientsMutex.Unlock()

	return len(controller.clients)
}

func (controller *ServerLogicController) snapshotClients() map[string]*model.Client {
	controller.clientsMutex.Lock()
	defer controller.clientsMutex.Unlock()

	clients := make(map[string]*model.Client, len(controller.clients))
	for id, client := range controller.clients {
		clients[id] = client
	}

	return clients
}

func (controller *ServerLogicController) registerClient(client *model.Client) *model.Client {
	controller.clientsMutex.Lock()
	defer controller.clientsMutex.Unlock()

	replaced := controller.clients[client.Id]
	controller.clients[client.Id] = client

	return replaced
}

func (controller *ServerLogicController) unregisterClient(client *model.Client) bool {
	controller.clientsMutex.Lock()
	defer controller.clientsMutex.Unlock()

	current, ok := controller.clients[client.Id]
	if !ok || current != client {
		return false
	}

	delete(controller.clients, client.Id)
	return true
}

func (session *trainingSession) reset(clients map[string]*model.Client) uuid.UUID {
	session.mu.Lock()
	defer session.mu.Unlock()

	session.trainingClientIDs = make(map[string]struct{}, len(clients))
	for id := range clients {
		session.trainingClientIDs[id] = struct{}{}
	}

	session.completedClients = make(map[string]struct{}, len(clients))
	session.errorMessage = nil
	session.prepID = uuid.New()

	return session.prepID
}

func (session *trainingSession) trackClientDone(clientID string, prepID string) bool {
	session.mu.Lock()
	defer session.mu.Unlock()

	if prepID != session.prepID.String() {
		return false
	}

	if _, ok := session.trainingClientIDs[clientID]; !ok {
		return false
	}

	if _, ok := session.completedClients[clientID]; ok {
		return false
	}

	session.completedClients[clientID] = struct{}{}
	slog.Info("Progress:", "done", len(session.completedClients), "total", len(session.trainingClientIDs))

	if len(session.completedClients) >= len(session.trainingClientIDs) {
		session.doneCond.Broadcast()
	}

	return true
}

func (session *trainingSession) reportError(message *model.GeneralMessage) {
	session.mu.Lock()
	defer session.mu.Unlock()

	session.errorMessage = message
	session.doneCond.Broadcast()
}

func (session *trainingSession) handleClientDisconnect(client *model.Client) {
	session.mu.Lock()
	defer session.mu.Unlock()

	if _, ok := session.trainingClientIDs[client.Id]; !ok {
		return
	}

	slog.Info("Aborting training if in preparation phase.")
	session.errorMessage = &model.GeneralMessage{
		Type:    "ERROR",
		Origin:  "server",
		Message: "Client '" + client.Id + "' disconnected while preparing.",
	}
	session.doneCond.Broadcast()
}

func (session *trainingSession) waitForPreparation() *model.GeneralMessage {
	session.mu.Lock()
	defer session.mu.Unlock()

	for len(session.completedClients) < len(session.trainingClientIDs) && session.errorMessage == nil {
		session.doneCond.Wait()
	}

	return session.errorMessage
}

func (controller *ServerLogicController) Route() {
	http.HandleFunc("/ws", controller.handleWS)
	http.HandleFunc("/start", controller.withCORS(controller.start))
	http.HandleFunc("/status", controller.withCORS(controller.status))
	http.HandleFunc("/clients", controller.withCORS(controller.connectedClients))
}

func (controller *ServerLogicController) withCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next(w, r)
	}
}

func (controller *ServerLogicController) handleWS(w http.ResponseWriter, r *http.Request) {
	// Get client ID from HTTP header (no need to check for duplicates since client ID = node name)
	clientId := r.Header.Get("X-Client-ID")

	conn, err := controller.upgrader.Upgrade(w, r, nil)
	exception.PanicLogging(err, "Unable to upgrade connection to WebSocket")

	// Get client address
	clientAddress := conn.RemoteAddr().String()
	slog.Info("Client connected:", "ID", clientId, "address", clientAddress)

	client := &model.Client{Conn: conn, Id: clientId, Address: clientAddress}
	replacedClient := controller.registerClient(client)
	if replacedClient != nil {
		slog.Warn("Replacing existing client connection", "ID", clientId)
		replacedClient.Conn.Close()
	}

	defer func() {
		conn.Close()
		if !controller.unregisterClient(client) {
			return
		}

		slog.Info("Client disconnected:", "ID", clientId, "address", clientAddress)
		session := controller.currentTrainingSession()
		if session != nil {
			session.handleClientDisconnect(client)
		}
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
			prepID := strings.TrimPrefix(generalMessage.Message, "DONE:")
			session := controller.currentTrainingSession()
			if session == nil || !session.trackClientDone(clientId, prepID) {
				slog.Info("Ignoring DONE message", "client", clientId, "prepId", prepID)
				continue
			}

			slog.Info("Reported DONE by", "client", clientId)
		}

		// Handle ERROR message
		if generalMessage.Type == "ERROR" {
			session := controller.currentTrainingSession()
			if session == nil {
				continue
			}

			expectedPrepID := session.prepID.String()
			if generalMessage.PrepID != "" && generalMessage.PrepID != expectedPrepID {
				slog.Warn("Ignoring ERROR message for stale preparation", "clientId", clientId, "prepId", generalMessage.PrepID, "expectedPrepId", expectedPrepID)
				continue
			}

			slog.Error("ERROR received from", "clientId", clientId, "prepId", generalMessage.PrepID)
			session.reportError(&generalMessage)
		}
	}
}

func (controller *ServerLogicController) start(w http.ResponseWriter, r *http.Request) {
	slog.Info("Received /start request")
	releaseTraining := false

	// Abort if there is an ongoing training run / client preparations
	session, ok := controller.reserveTrainingSession()
	if !ok {
		slog.Error("Aborting training due to already ongoing training run")

		// Get Flower details about current run to include in response
		runId := controller.getLatestTrainingRunId()
		flowerStr, err := controller.ServerLogicService.GetTrainingStatus(runId)
		if err != nil {
			slog.Error("Unable to get ongoing training status", "error", err)

			response := model.GeneralResponse{
				Message: "Aborting training due to already ongoing training run",
				Data: map[string]interface{}{
					"run-id": runId,
					"status": "unavailable",
					"error":  err.Error(),
				},
			}
			jsonData, _ := json.Marshal(response)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "60")
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write(jsonData)

			return
		}

		var flowerOut model.FlowerResponse
		json.Unmarshal([]byte(flowerStr), &flowerOut)
		var responseData interface{} = map[string]interface{}{
			"run-id": runId,
			"status": "unknown",
		}
		if len(flowerOut.Runs) > 0 {
			responseData = flowerOut.Runs[0]
		}

		response := model.GeneralResponse{
			Message: "Aborting training due to already ongoing training run",
			Data:    responseData,
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "60")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write(jsonData)

		return
	}
	releaseTraining = true
	defer func() {
		if releaseTraining {
			controller.finishTrainingSession(session)
		}
	}()

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
	clients := controller.snapshotClients()
	clientCount := len(clients)
	if clientCount < scenario.Strategy.MinAvailableClients {
		slog.Error("Aborting training due to insufficient client connections", "current", clientCount, "required", scenario.Strategy.MinAvailableClients)

		response := model.GeneralResponse{
			Message: fmt.Sprintf("Aborting training due to insufficient client connections (%d out of %d)", clientCount, scenario.Strategy.MinAvailableClients),
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
		for _, client := range clients {
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
				return
			}
		}
	}

	// Download datasets (if not present) if centralised property is set true
	if scenario.Centralised {
		if scenario.Dataset != "" {
			controller.MinioService.CreateDataset(scenario.Dataset)
		} else { // If "dataset" isn't set, "datasets" must be set (part of scenario validation)
			for _, dataset := range scenario.Datasets {
				controller.MinioService.CreateDataset(dataset)
			}
		}
	}

	// Download hold-out dataset (if not present) if hold_out_dataset property is set
	if scenario.HoldOutDataset != "" {
		controller.MinioService.CreateAndShareHoldOutDataset(scenario.HoldOutDataset)
	}

	// Set new preparation ID and reset run state.
	prepID := session.reset(clients)

	// Set config for Flower serverapp
	controller.ServerLogicService.SetFlowerConfig(scenario)

	// Send scenario to clients with uuid
	slog.Info("start", "totalClients", len(clients))
	controller.ServerLogicService.StartClients(scenario, prepID, clients)

	// Wait until all expected clients report DONE or an ERROR message is received.
	errorMessage := session.waitForPreparation()

	// Abort on error
	if errorMessage != nil {
		slog.Error("Aborting training due to error", "origin", errorMessage.Origin, "message", errorMessage.Message)

		response := model.GeneralResponse{
			Message: "Training run aborted due to client error",
			Data:    errorMessage,
		}
		jsonData, _ := json.Marshal(response)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write(jsonData)
		return
	}

	// Start Flower training if no error message
	slog.Info("All clients reported done")
	jsonStr, err := controller.ServerLogicService.StartTraining()
	if err != nil {
		slog.Error("Unable to start Flower training", "error", err)

		response := model.GeneralResponse{
			Message: "Unable to start Flower training",
			Data:    err.Error(),
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write(jsonData)
		return
	}

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
	releaseTraining = false
	go func() {
		// Watch for creation of safetensors file in shared folder and upload to datastore
		controller.ServerLogicService.WatchForEndOfTraining(controller.MinioService.UploadArtifacts, scenario.Name)

		// Empty shared folder
		controller.ServerLogicService.RemoveArtifacts()

		// Free to start another training run
		controller.finishTrainingSession(session)
	}()
}

func (controller *ServerLogicController) status(w http.ResponseWriter, r *http.Request) {
	slog.Info("Received /status request with", "params", r.URL.Query())

	runId := r.URL.Query().Get("runId")
	jsonStr, err := controller.ServerLogicService.GetTrainingStatus(runId)
	if err != nil {
		slog.Error("Unable to get training status", "error", err)

		response := model.GeneralResponse{
			Message: "Flower status unavailable",
			Data: map[string]interface{}{
				"success": false,
				"runs":    []interface{}{},
				"error":   err.Error(),
			},
		}
		jsonData, _ := json.Marshal(response)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write(jsonData)
		return
	}

	// Response
	var flowerResponse model.FlowerResponse
	err = json.Unmarshal([]byte(jsonStr), &flowerResponse)
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
		copiedClient := model.Client{
			Conn:    nil, // Don't try to return Conn
			Id:      controller.clients[key].Id,
			Address: controller.clients[key].Address,
		}

		clientList = append(clientList, &copiedClient)
	}

	controller.clientsMutex.Unlock()

	// Response
	response := model.GeneralResponse{
		Message: "Following clients are connected to the server",
		Data:    clientList,
	}
	jsonData, _ := json.Marshal(response)
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)

}
