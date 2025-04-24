package controller

import (
	"context"
	"log"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type PreprocessorController struct {
	service.PreprocessorService
	service.MinioService
	service.KafkaService
	service.ParquetService
}

func NewPreprocessorController(preprocessorService *service.PreprocessorService, minioService *service.MinioService, kafkaService *service.KafkaService, parquetService *service.ParquetService) *PreprocessorController {
	return &PreprocessorController{
		PreprocessorService: *preprocessorService,
		MinioService:        *minioService,
		KafkaService:        *kafkaService,
		ParquetService:      *parquetService,
	}
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

	buf := controller.MinioService.LoadDataset(scenario.Dataset)
	rows := controller.ParquetService.ExtractRows(buf)

	for _, r := range rows {
		record := r.(map[string]interface{})
		log.Printf("Row: %+v", record)
	}
}
