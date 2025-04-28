package main

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/controller"
	repository "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository/impl"
	service "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service/impl"
)

func main() {
	// setup configuration
	config := configuration.New()
	minioClient := configuration.NewMinioClient(config)
	kafkaPrepareReader := configuration.NewKafkaPrepareReader(config)
	kafkaPreparedWriter := configuration.NewKafkaPreparedWriter(config)

	// repository
	minioRepository := repository.NewMinioRepositoryImpl(minioClient, config)
	localRepository := repository.NewLocalRepositoryImpl(config)
	kafkaRepository := repository.NewKafkaPublishRepositoryImpl(kafkaPrepareReader, kafkaPreparedWriter)
	defer kafkaRepository.Close()

	// service
	minioService := service.NewMinioServiceImpl(&minioRepository, &localRepository)
	kafkaService := service.NewKafkaServiceImpl(&kafkaRepository)

	// controller
	minioController := controller.NewMinioController(&minioService, &kafkaService, config)

	// consume
	minioController.StartConsuming(context.Background())
}
