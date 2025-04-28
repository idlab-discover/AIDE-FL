package controller

import (
	"encoding/json"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/service"
	"github.com/gofiber/fiber/v3"
)

type OrchestratorController struct {
	service.KafkaService
}

func NewOrchestratorController(kafkaService *service.KafkaService) *OrchestratorController {
	return &OrchestratorController{KafkaService: *kafkaService}
}

func (controller OrchestratorController) Route(app *fiber.App) {
	app.Post("/start", controller.Start)
}

func (controller OrchestratorController) Start(c fiber.Ctx) error {
	var scenario model.Scenario
	err := json.Unmarshal(c.Body(), &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	controller.KafkaService.PublishPrepareScenario(scenario)

	scenario.Status = "not preprocessed"
	return c.Status(fiber.StatusCreated).JSON(model.GeneralResponse{
		Message: "Success",
		Data:    scenario,
	})
}
