package impl

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service"
	"github.com/fasthttp/websocket"
	"github.com/fsnotify/fsnotify"
	"github.com/go-playground/validator/v10"
	"gopkg.in/ini.v1"
)

type serverLogicServiceImpl struct {
	configuration.Config
	*validator.Validate
}

func NewServerLogicServiceImpl(config configuration.Config) service.ServerLogicService {
	return &serverLogicServiceImpl{
		Config:   config,
		Validate: validator.New(),
	}
}

func (service *serverLogicServiceImpl) DecodeScenario(r io.ReadCloser) (*model.Scenario, error) {
	defer r.Close()

	var scenario model.Scenario
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	err := dec.Decode(&scenario)
	if err != nil {
		return nil, fmt.Errorf("unable to decode JSON to scenario: %w", err)
	}

	return &scenario, nil
}

func (service *serverLogicServiceImpl) ValidateScenario(scenario *model.Scenario) error {
	err := service.Validate.Struct(scenario)
	return err
}

func (service *serverLogicServiceImpl) StartClients(scenario *model.Scenario, clients map[string]*model.Client) {
	slog.Info("Starting", "clients", slices.Collect(maps.Keys(clients)))

	scenario_json, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario into JSON")

	for _, c := range clients {
		err := c.Conn.WriteMessage(websocket.TextMessage, scenario_json)
		exception.PanicLogging(err, "Unable to send scenario to", "client", c.Id)
	}
	slog.Info("Sent scenario to", "#clients", len(clients))
}

func (service *serverLogicServiceImpl) SetFlowerConfig(scenario *model.Scenario) {
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

func (service *serverLogicServiceImpl) GetTrainingStatus(runId string) string {
	slog.Info("Getting training status for", "runId", runId)
	resp, err := http.Get("http://localhost:8080/ls?runId=" + runId)
	exception.PanicLogging(err, "Unable to get training status")
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	exception.PanicLogging(err, "Unable to read body")

	bodyString := string(bodyBytes)
	return bodyString
}

func (service *serverLogicServiceImpl) WatchForSafetensors(callback func(string)) {
	slog.Info("Watching for creation of safetensors file...")

	watcher, err := fsnotify.NewWatcher()
	exception.PanicLogging(err, "Unable to create new watcher")
	defer watcher.Close()

	// Add directory to watch to watcher
	err = watcher.Add(path.Join("app", "share"))
	exception.PanicLogging(err, "Unable to add shared folder to watcher")

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}

			// Only interested in create events
			if event.Op&(fsnotify.Create) != 0 {
				if strings.HasSuffix(event.Name, ".safetensors") {
					// Wait to ensure file is fully written
					slog.Info("Safetensors file found", "file", event.Name)
					time.Sleep(1 * time.Second)
					callback(event.Name)
					return
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			slog.Error("Watcher", "error", err)
		}
	}
}
