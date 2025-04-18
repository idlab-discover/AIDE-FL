package controller

import (
	"encoding/json"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/exception"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
	"github.com/gofiber/fiber/v3"
)

type PreprocessorController struct {
	service.PreprocessorService
}

func NewPreprocessorController(preprocessorService *service.PreprocessorService) *PreprocessorController {
	return &PreprocessorController{PreprocessorService: *preprocessorService}
}

func (controller PreprocessorController) Route(app *fiber.App) {
	app.Post("/preprocess", controller.Preprocess)
}

func (controller PreprocessorController) Preprocess(c fiber.Ctx) error {
	var scenario model.Scenario
	err := json.Unmarshal(c.Body(), &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	return c.Status(fiber.StatusAccepted).JSON(model.GeneralResponse{
		Message: "Success",
		Data:    "Not implemented",
	})
}
