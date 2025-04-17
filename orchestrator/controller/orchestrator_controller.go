package controller

import (
	"encoding/json"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/gofiber/fiber/v3"
)

type OrchestratorController struct {
}

func NewOrchestratorController() *OrchestratorController {
	return &OrchestratorController{}
}

func (controller OrchestratorController) Route(app *fiber.App) {
	app.Post("/scenario", controller.Start)
}

func (controller OrchestratorController) Start(c fiber.Ctx) error {
	var request model.Scenario
	err := json.Unmarshal(c.Body(), &request)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	status := model.Status{Status: "Started"}
	return c.Status(fiber.StatusCreated).JSON(model.GeneralResponse{
		Message: "Success",
		Data:    status,
	})
}
