package main

import (
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/configuration"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/controller"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/exception"
	service "github.com/MartienskieNum1/AIDE-FL/preprocessor/service/impl"
	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
)

func main() {
	// setup configuration
	config := configuration.New()

	// repository

	// service
	preprocessorService := service.NewPrepocessorServiceImpl()

	// controller
	preprocessorController := controller.NewPreprocessorController(&preprocessorService)

	// setup fiber
	app := fiber.New()
	app.Use(logger.New())

	// routing
	preprocessorController.Route(app)

	// start app
	err := app.Listen(config.Get("LISTEN_PORT"))
	exception.PanicLogging(err, "Unable to start Fiber app")
}
