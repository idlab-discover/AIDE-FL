package controller

import (
	"context"
	"encoding/json"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/service"
	"github.com/gofiber/fiber/v3"
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

func (controller OrchestratorController) Route(app *fiber.App) {
	app.Post("/start", controller.Start)
}

func (controller OrchestratorController) Start(c fiber.Ctx) error {
	var scenario model.Scenario
	err := json.Unmarshal(c.Body(), &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	go controller.handleScenarioFlow(scenario)

	return c.Status(fiber.StatusCreated).JSON(model.GeneralResponse{
		Message: "Started",
		Data:    scenario,
	})
}

func (controller *OrchestratorController) handleScenarioFlow(scenario model.Scenario) {
	// Send scenario to Minio proxy for preparation
	controller.KafkaService.PublishPrepareScenario(scenario)
	// Receive scenario with status from Minio proxy
	scenario = controller.KafkaService.ConsumePreparedScenario(context.Background())

	// Send scenario to preprocessor if preprocessing is needed according to scenario status
	if !controller.OrchestratorService.IsPreprocessed(scenario) {
		controller.KafkaService.PublishPreprocessScenario(scenario)

		scenario = controller.KafkaService.ConsumePreprocessedScenario(context.Background()) // Receive scenario when preprocessed
	}

	// Send scenario to trainer
	controller.KafkaService.PublishTrainScenario(scenario)

	// Receive scenario from trainer when trained
	scenario = controller.KafkaService.ConsumeTrainedScenario(context.Background())
}
