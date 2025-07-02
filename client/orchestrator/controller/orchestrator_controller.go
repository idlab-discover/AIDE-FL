package controller

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/url"

	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/service"
	"github.com/gorilla/websocket"
)

type OrchestratorController struct {
	service.OrchestratorService
	service.KafkaService
}

func NewOrchestratorController(orchestratorService *service.OrchestratorService, kafkaService *service.KafkaService) *OrchestratorController {
	return &OrchestratorController{
		OrchestratorService: *orchestratorService,
		KafkaService:        *kafkaService,
	}
}

func (controller OrchestratorController) ConnectServer(host string) {
	u := url.URL{Scheme: "ws", Host: host, Path: "/ws"}

	// TODO: Set retries
	c, _, err := websocket.DefaultDialer.Dial(u.String(), nil)
	exception.PanicLogging(err, "Unable to connect to server")
	defer c.Close()
	slog.Info("Connected to", "url", u.String())

	// Start general message listener goroutine
	go controller.listenForGeneralMessages(c)

	for {
		_, message, err := c.ReadMessage()
		exception.PanicLogging(err, "Unable to read message")

		// Suppose message is a scenario in JSON format
		var scenario model.Scenario
		err = json.Unmarshal(message, &scenario)
		exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

		controller.handleScenarioFlow(scenario)

		slog.Info("Sending 'done' to server")
		err = c.WriteMessage(websocket.TextMessage, []byte("done"))
		exception.PanicLogging(err, "Unable to send 'done' message")
	}
}

func (controller *OrchestratorController) handleScenarioFlow(scenario model.Scenario) {

	// Send scenario to datastore-logic for preparation
	controller.KafkaService.PublishPrepareScenario(scenario)

	// Receive scenario with status from datastore-logic
	scenario = controller.KafkaService.ConsumePreparedScenario(context.Background())

	// Send scenario to preprocessor
	controller.KafkaService.PublishPreprocessScenario(scenario)

	// Receive scenario from preprocessor
	scenario = controller.KafkaService.ConsumePreprocessedScenario(context.Background())

	// Send scenario to trainer
	controller.KafkaService.PublishTrainScenario(scenario)

	// Receive scenario from trainer when trained
	scenario = controller.KafkaService.ConsumeTrainedScenario(context.Background())
}

func (controller *OrchestratorController) listenForGeneralMessages(conn *websocket.Conn) {
	for {
		m := controller.KafkaService.ConsumeGeneralMessage(context.Background())

		byteArray, err := json.Marshal(m)
		exception.PanicLogging(err, "Unable to marshal general message to JSON")

		// Send general message to the server via WebSocket
		err = conn.WriteMessage(websocket.TextMessage, byteArray)
		if err != nil {
			slog.Error("Unable to send general message to server", "err", err)
		}
	}
}
