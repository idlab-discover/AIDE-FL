package impl

import (
	"encoding/json"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"os"
	"path"
	"slices"

	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server-logic/service"
	"github.com/fasthttp/websocket"
	"gopkg.in/ini.v1"
)

type serverLogicServiceImpl struct {
	configuration.Config
}

func NewServerLogicServiceImpl(config configuration.Config) service.ServerLogicService {
	return &serverLogicServiceImpl{Config: config}
}

func (service *serverLogicServiceImpl) StartClients(scenario model.Scenario, clients map[string]*model.Client) {
	slog.Info("Starting", "clients", slices.Collect(maps.Keys(clients)))

	scenario_json, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario into JSON")

	for _, c := range clients {
		err := c.Conn.WriteMessage(websocket.TextMessage, scenario_json)
		exception.PanicLogging(err, "Unable to send scenario to", "client", c.Id)
	}
	slog.Info("Sent scenario to", "#clients", len(clients))
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

func (service *serverLogicServiceImpl) StartTraining() string {
	slog.Info("Sending training trigger to Flower serverapp")
	resp, err := http.Post("http://localhost:8080/run", "application/json", nil)
	exception.PanicLogging(err, "Unable to send trigger to start training")
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	exception.PanicLogging(err, "Unable to read body")

	bodyString := string(bodyBytes)
	return bodyString
}
