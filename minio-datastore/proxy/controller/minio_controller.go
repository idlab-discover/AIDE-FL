package controller

import (
	"context"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service"
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

func (controller MinioController) StartConsuming(ctx context.Context) {
	controller.KafkaService.ConsumePrepareScenario(ctx, controller.Prepare)
}

func (controller MinioController) Prepare(scenario model.Scenario) {
	slog.Info("Preparing", "scenario", scenario.Name)

	// The MinioService returns scenario with a Status
	scenario = controller.MinioService.Create(scenario)

	controller.KafkaService.PublishPreparedScenario(scenario)
}
