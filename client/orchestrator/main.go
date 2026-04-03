package main

import (
	"github.com/Str-Gen/AIDE-FL/client/orchestrator/configuration"
	"github.com/Str-Gen/AIDE-FL/client/orchestrator/controller"
	service "github.com/Str-Gen/AIDE-FL/client/orchestrator/service/impl"
)

func main() {
	// setup configuration
	config := configuration.New()
	kafkaFactory := configuration.NewKafkaFactory(config)

	// repository
	kafkaPrepareWriterRepo := kafkaFactory.NewKafkaWriterRepository(config.Get("KAFKA_PREPARE_TOPIC"))
	defer kafkaPrepareWriterRepo.Close()
	kafkaPreparedReaderRepo := kafkaFactory.NewKafkaReaderRepository(config.Get("KAFKA_PREPARED_TOPIC"))
	defer kafkaPreparedReaderRepo.Close()
	kafkaPreprocessWriterRepo := kafkaFactory.NewKafkaWriterRepository(config.Get("KAFKA_PREPROCESS_TOPIC"))
	defer kafkaPreprocessWriterRepo.Close()
	kafkaPreprocessedReaderRepo := kafkaFactory.NewKafkaReaderRepository(config.Get("KAFKA_PREPROCESSED_TOPIC"))
	defer kafkaPreprocessedReaderRepo.Close()
	kafkaTrainWriterRepo := kafkaFactory.NewKafkaWriterRepository(config.Get("KAFKA_TRAIN_TOPIC"))
	defer kafkaTrainWriterRepo.Close()
	kafkaTrainedReaderRepo := kafkaFactory.NewKafkaReaderRepository(config.Get("KAFKA_TRAINED_TOPIC"))
	defer kafkaTrainedReaderRepo.Close()
	kafkaGeneralMessageReaderRepo := kafkaFactory.NewKafkaReaderRepository(config.Get("KAFKA_MESSAGE_TOPIC"))
	defer kafkaGeneralMessageReaderRepo.Close()

	// service
	orchestratorService := service.NewOrchestratorServiceImpl()
	kafkaService := service.NewKafkaServiceImpl(
		&kafkaPrepareWriterRepo,
		&kafkaPreparedReaderRepo,
		&kafkaPreprocessWriterRepo,
		&kafkaPreprocessedReaderRepo,
		&kafkaTrainWriterRepo,
		&kafkaTrainedReaderRepo,
		&kafkaGeneralMessageReaderRepo,
	)

	// controller
	orchestratorController := controller.NewOrchestratorController(&orchestratorService, &kafkaService)

	// start
	orchestratorController.ConnectServer(config.Get("SERVER_IP")+config.Get("SERVER_PORT"), config.Get("NODE_NAME"))
}
