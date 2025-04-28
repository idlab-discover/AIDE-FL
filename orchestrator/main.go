package main

import (
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/configuration"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/controller"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	repository "github.com/MartienskieNum1/AIDE-FL/orchestrator/repository/impl"
	service "github.com/MartienskieNum1/AIDE-FL/orchestrator/service/impl"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
)

func main() {
	// setup configuration
	config := configuration.New()
	kafkaPrepareWriter := configuration.NewKafkaPrepareWriter(config)

	// repository
	kafkaRepository := repository.NewKafkaRepositoryImpl(kafkaPrepareWriter)

	// service
	kafkaService := service.NewKafkaServiceImpl(kafkaRepository)

	// controller
	orchestratorController := controller.NewOrchestratorController(&kafkaService)

	// setup fiber
	app := fiber.New()
	app.Use(logger.New())

	// routing
	orchestratorController.Route(app)

	// start app
	err := app.Listen(config.Get("LISTEN_PORT"))
	exception.PanicLogging(err, "Unable to start Fiber app")
}
