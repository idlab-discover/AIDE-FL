package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/url"
	"time"

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
	maxRetries := 8
	baseDelay := time.Second

	for {
		var c *websocket.Conn
		var err error

		// Retry loop for (re)connection
		for i := 0; i < maxRetries; i++ {
			c, _, err = websocket.DefaultDialer.Dial(u.String(), nil)
			if err == nil {
				slog.Info("Connected to", "url", u.String())
				break
			}
			slog.Warn("Failed to connect", "attempt", i+1, "error", err)
			time.Sleep(baseDelay * time.Duration(1<<i)) // exponential backoff
		}

		if err != nil {
			slog.Error("Unable to connect to server after retries", "error", err)
			time.Sleep(time.Minute) // back off before trying again
			continue
		}

		// Start general message listener goroutine
		ctx, cancel := context.WithCancel(context.Background())
		go controller.consumeGeneralMessages(ctx, c)

		// Listen for incoming scenario from server
		err = controller.listenForScenarios(c)

		// Cleanup
		slog.Warn("Disconnected from server, retrying", "error", err)
		slog.Info("Stopping general message consumer")
		cancel()  // Stop consuming general messages
		c.Close() // Close connection before trying to reconnect

		time.Sleep(time.Second * 2)
	}
}

func (controller *OrchestratorController) consumeGeneralMessages(ctx context.Context, conn *websocket.Conn) {
	for {
		m, err := controller.KafkaService.ConsumeGeneralMessage(ctx)
		if err != nil {
			slog.Error("Unable to consume general message", "error", err)
			return
		}

		byteArray, err := json.Marshal(m)
		if err != nil {
			slog.Error("Unable to marshal general message to JSON", "error", err)
			continue // Skip bad message
		}

		// Send general message to the server via WebSocket
		err = conn.WriteMessage(websocket.TextMessage, byteArray)
		if err != nil {
			slog.Error("Unable to send general message to server", "err", err)
		}
	}
}

func (controller *OrchestratorController) listenForScenarios(c *websocket.Conn) error {
	for {
		_, message, err := c.ReadMessage()
		if err != nil {
			return fmt.Errorf("unable to read message: %w", err)
		}

		// Suppose message is a scenario in JSON format
		var scenario model.Scenario
		err = json.Unmarshal(message, &scenario)
		if err != nil {
			slog.Error("Unable to unmarshal JSON to scenario", "error", err)
			continue // Skip bad message
		}

		controller.handleScenarioFlow(scenario)

		slog.Info("Sending 'done' to server")
		err = c.WriteMessage(websocket.TextMessage, []byte("done"))
		if err != nil {
			return fmt.Errorf("unable to send 'done': %w", err)
		}
	}
}

func (controller *OrchestratorController) handleScenarioFlow(scenario model.Scenario) {

	// Send scenario to datastore-logic for preparation
	controller.KafkaService.PublishPrepareScenario(scenario)

	// Receive scenario with status from datastore-logic
	scenario, err := controller.KafkaService.ConsumePreparedScenario(context.Background())
	exception.PanicLogging(err, "Unable to consume prepared scenario")

	// Send scenario to preprocessor
	controller.KafkaService.PublishPreprocessScenario(scenario)

	// Receive scenario from preprocessor
	scenario, err = controller.KafkaService.ConsumePreprocessedScenario(context.Background())
	exception.PanicLogging(err, "Unable to consume preprocessed scenario")

	// Send scenario to trainer
	controller.KafkaService.PublishTrainScenario(scenario)

	// Receive scenario from trainer when trained
	scenario, err = controller.KafkaService.ConsumeTrainedScenario(context.Background())
	exception.PanicLogging(err, "Unable to consume trained scenario")
}
