package controller

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"

	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/service"
	"github.com/fasthttp/websocket"
)

type ServerLogicController struct {
	service.ServerLogicService
	upgrader     websocket.Upgrader
	clients      map[string]*model.Client
	clientsMutex *sync.Mutex
	doneCount    int
	totalClients int
	doneCond     *sync.Cond
}

func NewServerLogicController(serverLogicService *service.ServerLogicService) *ServerLogicController {
	return &ServerLogicController{
		ServerLogicService: *serverLogicService,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
		clients:      make(map[string]*model.Client),
		clientsMutex: &sync.Mutex{},
		doneCount:    0,
		totalClients: 0,
		doneCond:     sync.NewCond(&sync.Mutex{}),
	}
}

func (controller *ServerLogicController) Route() {
	http.HandleFunc("/ws", controller.handleWS)
	http.HandleFunc("/start", controller.start)
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
		Message: "Clients are done preparing, training has started",
		Data:    jsonFlower,
	}
	jsonData, _ := json.Marshal(response)
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)
}
