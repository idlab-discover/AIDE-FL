package impl

import (
	"encoding/json"
	"log/slog"
	"os"
	"path"
	"strings"

	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/service"
	"gopkg.in/ini.v1"

	"github.com/gofiber/fiber/v3/client"
)

type serverLogicServiceImpl struct {
	configuration.Config
}

func NewServerLogicServiceImpl(config configuration.Config) service.ServerLogicService {
	return &serverLogicServiceImpl{Config: config}
}

func (service *serverLogicServiceImpl) StartClients(scenario model.Scenario) {
	clients := strings.Split(service.Config.Get("CLIENTS"), ",")
	slog.Info("Starting", "clients", clients)

	cc := client.New()
	// TODO: parallelize
	for i := 0; i < len(clients); i++ {
		client_url := "http://" + clients[i] + service.Config.Get("CLIENT_LISTEN_PORT") + "/start"
		slog.Info("Sending POST to", "url", client_url)

		config := new(client.Config)
		config.Body = scenario // Client body serialises json by default
		_, err := cc.Post(client_url, *config)
		exception.PanicLogging(err, "Unable to send POST to", "client", clients[i])
	}
}

func (service *serverLogicServiceImpl) SetFlowerConfig(scenario model.Scenario) {
	slog.Info("Updating config file")
	config_path := path.Join("app", "share", "config.ini")

	scenario_json, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to unmarshal scenario from JSON")

	if _, err := os.Stat(config_path); err != nil {
		if os.IsNotExist(err) {
			slog.Info("Config file does not exist. Creating")
			file, err := os.Create(config_path)
			exception.PanicLogging(err, "Unable to create config file")
			defer file.Close()

		} else {
			exception.PanicLogging(err, "Unable to check config file")
		}
	}

	cfg, err := ini.LoadSources(ini.LoadOptions{}, config_path)
	exception.PanicLogging(err, "Unable to load config file")

	section, err := cfg.NewSection("serverapp")
	exception.PanicLogging(err, "Unable to create section")

	section.Key("changed").SetValue("1")
	section.Key("scenario").SetValue(string(scenario_json))

	err = cfg.SaveTo(config_path)
	exception.PanicLogging(err, "Unable to save config file")
}
