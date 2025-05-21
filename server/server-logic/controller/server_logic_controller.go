package controller

import (
	"encoding/json"

	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/service"
	"github.com/gofiber/fiber/v3"
)

type ServerLogicController struct {
	service.ServerLogicService
}

func NewServerLogicController(serverLogicService *service.ServerLogicService) *ServerLogicController {
	return &ServerLogicController{
		ServerLogicService: *serverLogicService,
	}
}

func (controller ServerLogicController) Route(app *fiber.App) {
	app.Post("/start", controller.Start)
}

func (controller ServerLogicController) Start(c fiber.Ctx) error {
	var scenario model.Scenario
	err := json.Unmarshal(c.Body(), &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	// Set config for Flower serverapp
	controller.ServerLogicService.SetFlowerConfig(scenario)

	// Send request to clients
	controller.ServerLogicService.StartClients(scenario)

	return c.Status(fiber.StatusCreated).JSON(model.GeneralResponse{
		Message: "Success",
		Data:    scenario,
	})
}
