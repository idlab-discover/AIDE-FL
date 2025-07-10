package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
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

	cancelScenarioFunc context.CancelFunc // This function can be called to stop scenario flow
}

func NewOrchestratorController(orchestratorService *service.OrchestratorService, kafkaService *service.KafkaService) *OrchestratorController {
	return &OrchestratorController{
		OrchestratorService: *orchestratorService,
		KafkaService:        *kafkaService,
	}
}

func (controller OrchestratorController) ConnectServer(host string, clientId string) {
	u := url.URL{Scheme: "ws", Host: host, Path: "/ws"}
	maxRetries := 7
	baseDelay := time.Second

	// Create custom header containing client identifier (node name)
	headers := http.Header{}
	headers.Set("X-Client-ID", clientId)

	for {
		var c *websocket.Conn
		var err error

		// Retry loop for (re)connection
		for i := 0; i < maxRetries; i++ {
			c, _, err = websocket.DefaultDialer.Dial(u.String(), headers)
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

		// Create context to cancel consuming general messages when connection is lost
		ctxGenMessages, cancelConsumingGeneralMessages := context.WithCancel(context.Background())

		// Start general message listener goroutine
		go controller.consumeGeneralMessages(ctxGenMessages, c)

		// Listen for incoming scenario from server
		err = controller.listenForScenarios(c)

		// Cleanup
		slog.Warn("Disconnected from server, retrying", "error", err)
		slog.Info("Stopping general message consumer")
		cancelConsumingGeneralMessages() // Stop consuming general messages
		c.Close()                        // Close connection before trying to reconnect

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

		if m.Type == "ERROR" && controller.cancelScenarioFunc != nil {
			slog.Warn("Cancelling scenario flow due to ERROR")
			controller.cancelScenarioFunc()
			controller.cancelScenarioFunc = nil
		}

		byteArray, err := json.Marshal(m)
		if err != nil {
			slog.Error("Unable to marshal general message to JSON", "error", err)
			continue // Skip bad message
		}

		// Send general message to the server via WebSocket
		slog.Info("Sending general message to server")
		err = conn.WriteMessage(websocket.TextMessage, byteArray)
		if err != nil {
			slog.Error("Unable to send general message to server", "err", err)
		}
	}
}

func (controller *OrchestratorController) listenForScenarios(c *websocket.Conn) error {
	for {
		slog.Info("Waiting for incoming scenario...")

		_, message, err := c.ReadMessage()
		if err != nil {
			return fmt.Errorf("unable to read message: %w", err)
		}

		// Suppose message is a scenario in JSON format
		var scenario model.Scenario
		err = json.Unmarshal(message, &scenario)
		if err != nil {
			slog.Error("Unable to unmarshal JSON to scenario. Skipping bad message.", "error", err)
			continue // Skip bad message
		}

		// Create context + cancel func per scenario
		handleScenarioCtx, cancelScenarioFlow := context.WithCancel(context.Background())
		controller.cancelScenarioFunc = cancelScenarioFlow

		controller.handleScenarioFlow(scenario, handleScenarioCtx)

		select {
		case <-handleScenarioCtx.Done():
			continue
		default:

			slog.Info("Sending DONE to server")
			m := model.GeneralMessage{
				Type:    "INFO",
				Origin:  "orchestrator",
				Message: "DONE",
			}
			byteArray, err := json.Marshal(m)
			exception.PanicLogging(err, "Unable to marshal general message to JSON")

			err = c.WriteMessage(websocket.TextMessage, byteArray)
			if err != nil {
				return fmt.Errorf("unable to send DONE: %w", err)
			}
		}
	}
}

func (controller *OrchestratorController) handleScenarioFlow(scenario model.Scenario, ctx context.Context) {

	// Send scenario to datastore-logic for preparation
	controller.KafkaService.PublishPrepareScenario(scenario)

	// Receive scenario with status from datastore-logic
	scenario, err := controller.KafkaService.ConsumePreparedScenario(ctx)
	if err != nil {
		slog.Error("Unable to consume prepared scenario. Cancelling scenario flow.", "error", err)
		return
	}

	// Send scenario to preprocessor
	controller.KafkaService.PublishPreprocessScenario(scenario)

	// Receive scenario from preprocessor
	scenario, err = controller.KafkaService.ConsumePreprocessedScenario(ctx)
	if err != nil {
		slog.Error("Unable to consume preprocessed scenario. Cancelling scenario flow.", "error", err)
		return
	}

	// Send scenario to trainer
	controller.KafkaService.PublishTrainScenario(scenario)

	// Receive scenario from trainer when trained
	scenario, err = controller.KafkaService.ConsumeTrainedScenario(ctx)
	if err != nil {
		slog.Error("Unable to consume trained scenario. Cancelling scenario flow.", "error", err)
		return
	}
}
