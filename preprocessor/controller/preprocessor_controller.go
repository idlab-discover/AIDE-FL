package controller

import (
	"context"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type PreprocessorController struct {
	service.PreprocessorService
	service.KafkaService
}

func NewPreprocessorController(preprocessorService *service.PreprocessorService, kafkaService *service.KafkaService) *PreprocessorController {
	return &PreprocessorController{PreprocessorService: *preprocessorService, KafkaService: *kafkaService}
}

func (controller PreprocessorController) StartConsumingScenarios(ctx context.Context) {
	controller.KafkaService.ConsumeScenarioStatusEvents(ctx, controller.Preprocess)
}

func (controller PreprocessorController) Preprocess(scenario model.Scenario) {
	slog.Info("Start preprocessing for", "scenario", scenario.Name)
}
