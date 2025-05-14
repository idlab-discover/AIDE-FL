package main

import (
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/configuration"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/controller"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	service "github.com/MartienskieNum1/AIDE-FL/orchestrator/service/impl"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
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

	// service
	orchestratorService := service.NewOrchestratorServiceImpl()
	kafkaService := service.NewKafkaServiceImpl(
		&kafkaPrepareWriterRepo,
		&kafkaPreparedReaderRepo,
		&kafkaPreprocessWriterRepo,
		&kafkaPreprocessedReaderRepo,
		&kafkaTrainWriterRepo,
		&kafkaTrainedReaderRepo,
	)

	// controller
	orchestratorController := controller.NewOrchestratorController(&orchestratorService, &kafkaService)

	// setup fiber
	app := fiber.New()
	app.Use(logger.New())

	// routing
	orchestratorController.Route(app)

	// start app
	err := app.Listen(config.Get("LISTEN_PORT"))
	exception.PanicLogging(err, "Unable to start Fiber app")
}
