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

	// repository
	kafkaConsumeRepository := repository.NewKafkaConsumeRepositoryImpl(config)
	defer kafkaConsumeRepository.Close()

	// service
	preprocessorService := service.NewPrepocessorServiceImpl()
	kafkaService := service.NewKafkaServiceImpl(kafkaConsumeRepository)

	// controller
	preprocessorController := controller.NewPreprocessorController(&preprocessorService, &kafkaService)

	// consume
	preprocessorController.StartConsumingScenarios(context.Background())
}
