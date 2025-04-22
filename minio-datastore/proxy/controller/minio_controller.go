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
	service.KafkaService
	configuration.Config
}

func NewMinioController(minioService *service.MinioService, kafkaService *service.KafkaService, config configuration.Config) *MinioController {
	this := &MinioController{MinioService: *minioService, KafkaService: *kafkaService, Config: config}

	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_BUCKET"))

	return this
}

func (controller MinioController) Route(app *fiber.App) {
	app.Post("/scenario", controller.Create)
}

func (controller MinioController) Create(c fiber.Ctx) error {
	var scenario model.Scenario
	err := json.Unmarshal(c.Body(), &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	// The MinioService returns scenario with Status
	scenario = controller.MinioService.Create(scenario)
	controller.KafkaService.PublishDatasetStatus(scenario)

	return c.Status(fiber.StatusCreated).JSON(model.GeneralResponse{
		Message: "Success",
		Data:    scenario,
	})
}
