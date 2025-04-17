package main

import (
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/controller"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/exception"
	repository "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository/impl"
	service "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service/impl"
	"github.com/gofiber/fiber/v3"
)

func main() {
	// setup configuration
	config := configuration.New()
	minioClient := configuration.NewMinioClient(config)

	// repository
	minioRepository := repository.NewMinioRepositoryImpl(minioClient, config)
	localRepository := repository.NewLocalRepositoryImpl(config)

	// service
	minioService := service.NewMinioServiceImpl(&minioRepository, &localRepository, config)

	// controller
	minioController := controller.NewMinioController(&minioService, config)

	// setup fiber
	app := fiber.New()

	// routing
	minioController.Route(app)

	// start app
	err := app.Listen(config.Get("LISTEN_PORT"))
	exception.PanicLogging(err, "Unable to start Fiber app")
}
