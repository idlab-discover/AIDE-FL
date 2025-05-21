package main

import (
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/controller"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/exception"
	service "github.com/MartienskieNum1/AIDE-FL/server/server-logic/service/impl"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
)

func main() {
	// setup configuration
	config := configuration.New()

	// repository

	// service
	serverLogicService := service.NewServerLogicServiceImpl(config)

	// controller
	orchestratorController := controller.NewServerLogicController(&serverLogicService)

	// setup fiber
	app := fiber.New()
	app.Use(logger.New())

	// routing
	orchestratorController.Route(app)

	// start app
	err := app.Listen(config.Get("LISTEN_PORT"))
	exception.PanicLogging(err, "Unable to start Fiber app")
}
