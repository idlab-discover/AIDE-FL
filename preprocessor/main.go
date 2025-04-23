package main

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/configuration"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/controller"
	repository "github.com/MartienskieNum1/AIDE-FL/preprocessor/repository/impl"
	service "github.com/MartienskieNum1/AIDE-FL/preprocessor/service/impl"
)

func main() {
	// setup configuration
	config := configuration.New()
	minioClient := configuration.NewMinioClient(config)

	// repository
	minioRepository := repository.NewMinioRepositoryImpl(minioClient, config)
	kafkaConsumeRepository := repository.NewKafkaConsumeRepositoryImpl(config)
	defer kafkaConsumeRepository.Close()

	// service
	preprocessorService := service.NewPrepocessorServiceImpl()
	minioService := service.NewMinioServiceImpl(&minioRepository)
	kafkaService := service.NewKafkaServiceImpl(kafkaConsumeRepository)

	// controller
	preprocessorController := controller.NewPreprocessorController(&preprocessorService, &minioService, &kafkaService)

	// consume
	preprocessorController.StartConsumingScenarios(context.Background())
}
