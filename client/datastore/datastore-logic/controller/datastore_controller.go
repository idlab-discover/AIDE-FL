package controller

import (
	"context"
	"log/slog"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/configuration"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/service"
)

type DatastoreController struct {
	service.MinioService
	service.KafkaService
	configuration.Config
}

func NewMinioController(minioService *service.MinioService, kafkaService *service.KafkaService, config configuration.Config) *DatastoreController {
	this := &DatastoreController{MinioService: *minioService, KafkaService: *kafkaService, Config: config}

	this.MinioService.EnsureBucketExists(this.Config.Get("MINIO_BUCKET"))

	return this
}

func (controller DatastoreController) StartConsuming(ctx context.Context) {
	controller.KafkaService.ConsumePrepareScenario(ctx, controller.Prepare)
}

func (controller DatastoreController) Prepare(scenario model.Scenario) error {
	slog.Info("Preparing", "scenario", scenario.Name)

	scenario, err := controller.MinioService.Create(scenario)
	if err != nil {
		return err
	}

	controller.KafkaService.PublishPreparedScenario(scenario)
	return nil
}
