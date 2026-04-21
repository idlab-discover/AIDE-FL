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
	"path/filepath"
	"slices"

	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/configuration"
	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/exception"
	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/model"
	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/service"
	"github.com/fasthttp/websocket"
	"github.com/fsnotify/fsnotify"
	"github.com/go-playground/validator/v10"
	"github.com/google/uuid"
	"gopkg.in/ini.v1"
)

var shareDir = "/app/share"

const defaultFramework = "tensorflow"
const defaultAggregationStrategy = "fedavg"
const medianAggregationStrategy = "fedmedian"
const proxAggregationStrategy = "fedprox"
const adamAggregationStrategy = "fedadam"
const adagradAggregationStrategy = "fedadagrad"
const yogiAggregationStrategy = "fedyogi"
const bulyanAggregationStrategy = "bulyan"
const defaultXGBoostAggregationStrategy = "fedxgbbagging"
const cyclicXGBoostAggregationStrategy = "fedxgbcyclic"

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
	if len(scenario.Datasets) > 0 && len(scenario.Datasets) < scenario.Aggregation.MinAvailableClients {
		sl.ReportError(scenario.Datasets, "Datasets", "datasets", "min_available_clients", fmt.Sprintf("need at least %d datasets, got %d", scenario.Aggregation.MinAvailableClients, len(scenario.Datasets)))
	}

	switch scenario.Framework {
	case defaultFramework:
		if scenario.TensorFlow == nil {
			sl.ReportError(scenario.TensorFlow, "TensorFlow", "tensorflow", "required", "")
		} else if len(scenario.TensorFlow.Model.Config) == 0 && scenario.TensorFlow.Model.ConfigFile == "" {
			sl.ReportError(scenario.TensorFlow.Model, "Model", "tensorflow.model", "required", "")
		}
	case "scikit_learn":
		if scenario.ScikitLearn == nil {
			sl.ReportError(scenario.ScikitLearn, "ScikitLearn", "scikit_learn", "required", "")
		} else {
			if scenario.ScikitLearn.Model.Kind != "logistic_regression" {
				sl.ReportError(scenario.ScikitLearn.Model.Kind, "Kind", "scikit_learn.model.kind", "unsupported", "")
			}
			if scenario.ScikitLearn.Model.NFeatures < 1 {
				sl.ReportError(scenario.ScikitLearn.Model.NFeatures, "NFeatures", "scikit_learn.model.n_features", "required", "")
			}
			if len(scenario.ScikitLearn.Model.Classes) < 2 {
				sl.ReportError(scenario.ScikitLearn.Model.Classes, "Classes", "scikit_learn.model.classes", "required", "")
			}
		}
	case "xgboost":
		if scenario.XGBoost == nil {
			sl.ReportError(scenario.XGBoost, "XGBoost", "xgboost", "required", "")
		} else {
			if scenario.XGBoost.Model.Kind != "booster" {
				sl.ReportError(scenario.XGBoost.Model.Kind, "Kind", "xgboost.model.kind", "unsupported", "")
			}
			if scenario.XGBoost.Model.NFeatures < 1 {
				sl.ReportError(scenario.XGBoost.Model.NFeatures, "NFeatures", "xgboost.model.n_features", "required", "")
			}
			objective, ok := scenario.XGBoost.Params["objective"].(string)
			if !ok || objective == "" {
				sl.ReportError(scenario.XGBoost.Params, "Params", "xgboost.params.objective", "required", "")
			} else if objective != "binary:logistic" {
				sl.ReportError(objective, "Objective", "xgboost.params.objective", "unsupported", "")
			}
		}
	default:
		sl.ReportError(scenario.Framework, "Framework", "framework", "unsupported", "")
	}

	if scenario.Framework != defaultFramework && scenario.TensorFlow != nil {
		sl.ReportError(scenario.TensorFlow, "TensorFlow", "tensorflow", "forbidden", "")
	}
	if scenario.Framework != "scikit_learn" && scenario.ScikitLearn != nil {
		sl.ReportError(scenario.ScikitLearn, "ScikitLearn", "scikit_learn", "forbidden", "")
	}
	if scenario.Framework != "xgboost" && scenario.XGBoost != nil {
		sl.ReportError(scenario.XGBoost, "XGBoost", "xgboost", "forbidden", "")
	}
	switch scenario.Framework {
	case "xgboost":
		if scenario.Aggregation.Strategy != defaultXGBoostAggregationStrategy &&
			scenario.Aggregation.Strategy != cyclicXGBoostAggregationStrategy {
			sl.ReportError(scenario.Aggregation.Strategy, "Strategy", "aggregation.strategy", "unsupported", "")
		}
	case "scikit_learn":
		if scenario.Aggregation.Strategy == proxAggregationStrategy {
			sl.ReportError(scenario.Aggregation.Strategy, "Strategy", "aggregation.strategy", "unsupported", "")
		}
		fallthrough
	default:
		if scenario.Aggregation.Strategy != defaultAggregationStrategy &&
			scenario.Aggregation.Strategy != medianAggregationStrategy &&
			scenario.Aggregation.Strategy != proxAggregationStrategy &&
			scenario.Aggregation.Strategy != adamAggregationStrategy &&
			scenario.Aggregation.Strategy != adagradAggregationStrategy &&
			scenario.Aggregation.Strategy != yogiAggregationStrategy &&
			scenario.Aggregation.Strategy != bulyanAggregationStrategy {
			sl.ReportError(scenario.Aggregation.Strategy, "Strategy", "aggregation.strategy", "unsupported", "")
		}
	}

	if scenario.Aggregation.Strategy == bulyanAggregationStrategy {
		requiredClients := 4*scenario.Aggregation.NumMaliciousNodes + 3
		if scenario.Aggregation.MinAvailableClients < requiredClients {
			sl.ReportError(
				scenario.Aggregation.MinAvailableClients,
				"MinAvailableClients",
				"aggregation.min_available_clients",
				"min_bulyan_clients",
				fmt.Sprintf("need at least %d clients for %d malicious nodes", requiredClients, scenario.Aggregation.NumMaliciousNodes),
			)
		}
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

	service.normalizeScenario(&scenario)
	if err := service.resolveScenarioModel(&scenario); err != nil {
		return nil, err
	}

	return &scenario, nil
}

func (service *serverLogicServiceImpl) warnDeprecated(field string, replacement string) {
	slog.Warn("Deprecated scenario field encountered", "field", field, "replacement", replacement)
}

func (service *serverLogicServiceImpl) normalizeScenario(scenario *model.Scenario) {
	if scenario.LegacyHyper != nil {
		service.warnDeprecated("hyper", "training")
		scenario.Training = *scenario.LegacyHyper
		scenario.LegacyHyper = nil
	}

	if scenario.LegacyStrategy != nil {
		service.warnDeprecated("strategy", "aggregation")
		scenario.Aggregation = model.Aggregation{
			Strategy:            defaultAggregationStrategy,
			FractionFit:         scenario.LegacyStrategy.FractionFit,
			FractionEvaluate:    scenario.LegacyStrategy.FractionEvaluate,
			MinAvailableClients: scenario.LegacyStrategy.MinAvailableClients,
		}
		scenario.LegacyStrategy = nil
	}

	if scenario.Framework != defaultFramework {
		return
	}

	if scenario.TensorFlow == nil {
		scenario.TensorFlow = &model.TensorFlowFramework{}
	}

	if len(scenario.LegacyModel) != 0 {
		service.warnDeprecated("model", "tensorflow.model")
		scenario.TensorFlow.Model.Config = scenario.LegacyModel
		scenario.LegacyModel = nil
	}

	if scenario.LegacyModelFile != "" {
		service.warnDeprecated("model_file", "tensorflow.model.config_file")
		scenario.TensorFlow.Model.ConfigFile = scenario.LegacyModelFile
		scenario.LegacyModelFile = ""
	}
}

func (service *serverLogicServiceImpl) resolveScenarioModel(scenario *model.Scenario) error {
	if scenario.Framework != defaultFramework || scenario.TensorFlow == nil {
		return nil
	}

	modelPath := scenario.TensorFlow.Model.ConfigFile
	if len(scenario.TensorFlow.Model.Config) != 0 || modelPath == "" {
		return nil
	}

	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("unable to read model config file %q: %w", modelPath, err)
	}

	var modelData map[string]interface{}
	if err := json.Unmarshal(modelBytes, &modelData); err != nil {
		return fmt.Errorf("unable to decode model config file %q: %w", modelPath, err)
	}

	scenario.TensorFlow.Model.Config = modelData
	return nil
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
	config_path := path.Join(shareDir, "config.ini")

	scenario_json, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to unmarshal scenario from JSON")

	err = os.MkdirAll(filepath.Dir(config_path), 0o755)
	exception.PanicLogging(err, "Unable to create config directory")

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

	cfg.DeleteSection("serverapp")
	section, err := cfg.NewSection("serverapp")
	exception.PanicLogging(err, "Unable to create section")

	section.Key("changed").SetValue("1")
	section.Key("scenario").SetValue(string(scenario_json))

	err = cfg.SaveTo(config_path)
	exception.PanicLogging(err, "Unable to save config file")

	donePath := path.Join(shareDir, "done")
	if err := os.Remove(donePath); err != nil && !os.IsNotExist(err) {
		exception.PanicLogging(err, "Unable to remove stale done marker")
	}
}

func (service *serverLogicServiceImpl) StartTraining() (string, error) {
	slog.Info("Sending training trigger to Flower serverapp")
	resp, err := http.Post("http://localhost:8080/run", "application/json", nil)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	bodyString := string(bodyBytes)
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("flower serverapp returned %s: %s", resp.Status, bodyString)
	}

	return bodyString, nil
}

func (service *serverLogicServiceImpl) GetTrainingStatus(runId string) (string, error) {
	slog.Info("Getting training status for", "runId", runId)
	resp, err := http.Get("http://localhost:8080/ls?runId=" + runId)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	bodyString := string(bodyBytes)
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("flower serverapp returned %s: %s", resp.Status, bodyString)
	}

	return bodyString, nil
}

func (service *serverLogicServiceImpl) WatchForEndOfTraining(callback func(string), scenario_name string) {
	slog.Info("Watching for creation of 'done' file...")

	watcher, err := fsnotify.NewWatcher()
	exception.PanicLogging(err, "Unable to create new watcher")
	defer watcher.Close()

	// Add directory to watch to watcher
	err = os.MkdirAll(shareDir, 0o755)
	exception.PanicLogging(err, "Unable to create shared folder")

	err = watcher.Add(shareDir)
	exception.PanicLogging(err, "Unable to add shared folder to watcher")

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}

			// Only interested in create events
			if event.Op&(fsnotify.Create) != 0 {
				if filepath.Base(event.Name) == "done" { // File created by Flower server is named 'done'
					slog.Info("'done' file found marking end of training")
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
	dir := shareDir

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
