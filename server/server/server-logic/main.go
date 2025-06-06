package main

import (
	"log/slog"
	"net/http"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/controller"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/exception"
	repository "github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/repository/impl"
	service "github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service/impl"
)

func main() {
	// setup configuration
	config := configuration.New()
	minioClient := configuration.NewMinioClient(config)

	// repository
	minioRepository := repository.NewMinioRepositoryImpl(minioClient, config)

	// service
	minioService := service.NewMinioServiceImpl(&minioRepository)
	serverLogicService := service.NewServerLogicServiceImpl(config)

	// controller
	orchestratorController := controller.NewServerLogicController(&serverLogicService, &minioService, config)

	// routing
	orchestratorController.Route()

	// start server
	slog.Info("Server listening on", "port", config.Get("LISTEN_PORT"))
	err := http.ListenAndServe(config.Get("LISTEN_PORT"), nil)
	exception.PanicLogging(err, "Unable to start server")
}
