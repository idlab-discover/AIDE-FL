package controller

import (
	"context"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type PreprocessorController struct {
	service.PreprocessorService
	service.MinioService
	service.KafkaService
}

func NewPreprocessorController(preprocessorService *service.PreprocessorService, minioService *service.MinioService, kafkaService *service.KafkaService) *PreprocessorController {
	return &PreprocessorController{PreprocessorService: *preprocessorService, MinioService: *minioService, KafkaService: *kafkaService}
}

func (controller PreprocessorController) StartConsumingScenarios(ctx context.Context) {
	controller.KafkaService.ConsumeScenarioStatusEvents(ctx, controller.Preprocess)
}

func (controller PreprocessorController) Preprocess(scenario model.Scenario) {
	if scenario.Status == "preprocessed" {
		slog.Info("Already preprocessed", "scenario", scenario.Name)
		return
	}

	slog.Info("Start preprocessing", "scenario", scenario.Name)

}
