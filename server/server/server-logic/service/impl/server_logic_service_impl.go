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
	"github.com/google/uuid"
	"gopkg.in/ini.v1"
)

type serverLogicServiceImpl struct {
	configuration.Config
	*validator.Validate
}

func NewServerLogicServiceImpl(config configuration.Config) service.ServerLogicService {
	validate := validator.New()
	validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	return &serverLogicServiceImpl{
		Config:   config,
		Validate: validate,
	}
}

func datasetsValidation(sl validator.StructLevel) {
	scenario := sl.Current().Interface().(model.Scenario)

	// dataset and datasets cannot both be empty
	if scenario.Dataset == "" && len(scenario.Datasets) == 0 {
		sl.ReportError(scenario.Dataset, "Dataset", "dataset", "required_if_missing_datasets", "")
		sl.ReportError(scenario.Dataset, "Datasets", "datasets", "required_if_missing_dataset", "")
	}

	// dataset and datasets cannot both be filled
	if scenario.Dataset != "" && len(scenario.Datasets) > 0 {
		sl.ReportError(scenario.Dataset, "Dataset", "dataset", "mutually_exclusive", "")
		sl.ReportError(scenario.Datasets, "Datasets", "datasets", "mutually_exclusive", "")
	}

	// datasets must contain at least as much entries as the min amount of available clients that are required
	if len(scenario.Datasets) > 0 && len(scenario.Datasets) < scenario.Strategy.MinAvailableClients {
		sl.ReportError(scenario.Datasets, "Datasets", "datasets", "min_available_clients", fmt.Sprintf("need at least %d datasets, got %d", scenario.Strategy.MinAvailableClients, len(scenario.Datasets)))
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

func (service *serverLogicServiceImpl) scenarioForClient(baseScenario model.Scenario, clientId string) model.Scenario {
	// Use the dataset from Datasets if present
	if baseScenario.Dataset != "" {
		// Already a single dataset — nothing to do
		return baseScenario
	}

	// Use dataset for specified client
	dataset := baseScenario.Datasets[clientId]

	// Set only the relevant dataset
	baseScenario.Dataset = dataset
	baseScenario.Datasets = nil // remove the full map before sending

	return baseScenario
}

func (service *serverLogicServiceImpl) StartClients(scenario *model.Scenario, prepId uuid.UUID, clients map[string]*model.Client) {
	slog.Info("Starting", "clients", slices.Collect(maps.Keys(clients)))

	for _, c := range clients {
		clientScenario := service.scenarioForClient(*scenario, c.Id)

		scenarioMessage := model.ScenarioMessage{
			PrepId:   prepId.String(),
			Scenario: clientScenario,
		}

		scenarioJson, err := json.Marshal(scenarioMessage)
		exception.PanicLogging(err, "Unable to marshal scenario message into JSON")

		err = c.Conn.WriteMessage(websocket.TextMessage, scenarioJson)
		exception.PanicLogging(err, "Unable to send scenario message to", "client", c.Id)
	}
	slog.Info("Sent scenario message to", "#clients", len(clients))
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

func (service *serverLogicServiceImpl) WatchForSafetensors(callback func(string), scenario_name string) {
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
					callback(scenario_name)
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

func (service *serverLogicServiceImpl) RemoveArtifacts() {
	dir := path.Join("app", "share")

	entries, err := os.ReadDir(dir)
	if err != nil {
		panic(err)
	}

	for _, entry := range entries {
		entryPath := path.Join(dir, entry.Name())
		err := os.RemoveAll(entryPath) // removes file or recursively removes folder
		if err != nil {
			panic(err)
		}
	}

	fmt.Printf("Emptied folder: %s\n", dir)
}
