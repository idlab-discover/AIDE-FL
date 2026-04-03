package main

import (
	"context"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/configuration"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/controller"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/exception"
	repository "github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/repository/impl"
	service "github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/service/impl"
)

func main() {
	// setup configuration
	config := configuration.New()
	minioClient := configuration.NewMinioClient(config)
	kafkaPrepareReader := configuration.NewKafkaPrepareReader(config)
	kafkaPreparedWriter := configuration.NewKafkaPreparedWriter(config)
	kafkaGeneralMessageWriter := configuration.NewKafkaGeneralMessageWriter(config)

	// repository
	minioRepository := repository.NewMinioRepositoryImpl(minioClient, config)
	localRepository := repository.NewLocalRepositoryImpl(config)
	kafkaRepository := repository.NewKafkaRepositoryImpl(kafkaPrepareReader, kafkaPreparedWriter, kafkaGeneralMessageWriter)
	defer kafkaRepository.Close()

	// service
	minioService := service.NewMinioServiceImpl(&minioRepository, &localRepository)
	kafkaService := service.NewKafkaServiceImpl(&kafkaRepository)

	// setup exception
	exception.RegisterKafkaService(&kafkaService)

	// controller
	datastoreController := controller.NewMinioController(&minioService, &kafkaService, config)

	// consume
	datastoreController.StartConsuming(context.Background())
}
