package controller

import (
	"encoding/json"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service"
	"github.com/gofiber/fiber/v3"
)

type MinioController struct {
	service.MinioService
	configuration.Config
}

func NewMinioController(minioService *service.MinioService, config configuration.Config) *MinioController {
	this := &MinioController{MinioService: *minioService, Config: config}

	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_BUCKET"))

	return this
}

func (controller MinioController) Route(app *fiber.App) {
	app.Post("/scenario", controller.Create)
}

func (controller MinioController) Create(c fiber.Ctx) error {
	var request model.Scenario
	err := json.Unmarshal(c.Body(), &request)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	status := controller.MinioService.Create(c.Context(), request)
	return c.Status(fiber.StatusCreated).JSON(model.GeneralResponse{
		Message: "Success",
		Data:    status,
	})
}
