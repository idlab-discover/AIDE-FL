package main

import (
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/configuration"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/controller"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
)

func main() {
	// setup configuration
	config := configuration.New()

	// controller
	orchestratorController := controller.NewOrchestratorController()

	// setup fiber
	app := fiber.New()
	app.Use(logger.New())

	// routing
	orchestratorController.Route(app)

	// start app
	err := app.Listen(config.Get("LISTEN_PORT"))
	exception.PanicLogging(err, "Unable to start Fiber app")
}
