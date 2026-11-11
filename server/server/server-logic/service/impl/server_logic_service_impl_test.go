package impl

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-playground/validator/v10"
	"github.com/idlab-discover/AIDE-FL/server/server/server-logic/model"
	"gopkg.in/ini.v1"
)

func float64Ptr(v float64) *float64 {
	return &v
}

func TestDecodeScenarioResolvesLegacyModelFile(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "model.json")
	if err := os.WriteFile(modelPath, []byte(`{"class_name":"Sequential","config":{"layers":[]}}`), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	service := &serverLogicServiceImpl{}
	body := io.NopCloser(strings.NewReader(`{
		"name":"demo",
		"framework":"tensorflow",
		"dataset":"dataset",
		"label_column":"label",
		"preprocessing":{"drop_columns":[],"drop_rows":0,"shuffle":false,"split":{"train":0.7,"eval":0.2,"test":0.1}},
		"hyper":{"server_rounds":1,"local_epochs":1,"batch_size":16},
		"strategy":{"fraction_fit":1,"fraction_evaluate":1,"min_available_clients":1},
		"model_file":"` + modelPath + `"
	}`))

	scenario, err := service.DecodeScenario(body)
	if err != nil {
		t.Fatalf("DecodeScenario returned error: %v", err)
	}
	if scenario.Framework != defaultFramework {
		t.Fatalf("expected framework %q, got %q", defaultFramework, scenario.Framework)
	}
	if scenario.Training.ServerRounds != 1 {
		t.Fatalf("expected training to be normalized, got %#v", scenario.Training)
	}
	if scenario.Aggregation.Strategy != defaultAggregationStrategy {
		t.Fatalf("expected aggregation strategy %q, got %q", defaultAggregationStrategy, scenario.Aggregation.Strategy)
	}
	if scenario.TensorFlow == nil || len(scenario.TensorFlow.Model.Config) == 0 {
		t.Fatal("expected model to be loaded from model_file")
	}
}

func TestDecodeScenarioResolvesFrameworkAwareConfigFile(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "model.json")
	if err := os.WriteFile(modelPath, []byte(`{"class_name":"Sequential","config":{"layers":[]}}`), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	service := &serverLogicServiceImpl{}
	body := io.NopCloser(strings.NewReader(`{
		"name":"demo",
		"framework":"tensorflow",
		"dataset":"dataset",
		"label_column":"label",
		"preprocessing":{"drop_columns":[],"drop_rows":0,"shuffle":false,"split":{"train":0.7,"eval":0.2,"test":0.1}},
		"training":{"server_rounds":1,"local_epochs":1,"batch_size":16},
		"aggregation":{"strategy":"fedavg","fraction_fit":1,"fraction_evaluate":1,"min_available_clients":1},
		"tensorflow":{"model":{"config_file":"` + modelPath + `"}}
	}`))

	scenario, err := service.DecodeScenario(body)
	if err != nil {
		t.Fatalf("DecodeScenario returned error: %v", err)
	}
	if scenario.TensorFlow == nil {
		t.Fatal("expected tensorflow block to be present")
	}
	if scenario.TensorFlow.Model.ConfigFile != modelPath {
		t.Fatalf("expected config_file %q, got %q", modelPath, scenario.TensorFlow.Model.ConfigFile)
	}
	if len(scenario.TensorFlow.Model.Config) == 0 {
		t.Fatal("expected model config to be loaded from config_file")
	}
}

func TestValidateScenarioRequiresTensorFlowBlock(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     defaultFramework,
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation:   model.Aggregation{Strategy: defaultAggregationStrategy, FractionFit: 1, FractionEvaluate: 1, MinAvailableClients: 1},
	}

	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("expected validation error when model details are missing")
	}
}

func TestValidateScenarioAcceptsScikitLearnBlock(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "scikit_learn",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation:   model.Aggregation{Strategy: defaultAggregationStrategy, FractionFit: 1, FractionEvaluate: 1, MinAvailableClients: 1},
		ScikitLearn: &model.ScikitLearnFramework{
			Model: model.ScikitLearnModel{
				Kind:      "logistic_regression",
				NFeatures: 4,
				Classes:   []interface{}{0, 1},
			},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid scikit_learn scenario, got %v", err)
	}
}

func TestValidateScenarioAcceptsBulyanStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "scikit_learn",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            bulyanAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 7,
			NumMaliciousNodes:   1,
		},
		ScikitLearn: &model.ScikitLearnFramework{
			Model: model.ScikitLearnModel{
				Kind:      "logistic_regression",
				NFeatures: 4,
				Classes:   []interface{}{0, 1},
			},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid bulyan scenario, got %v", err)
	}
}

func TestValidateScenarioAcceptsFedMedianStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     defaultFramework,
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            medianAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid fedmedian scenario, got %v", err)
	}
}

func TestValidateScenarioAcceptsFedProxStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     defaultFramework,
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            proxAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
			ProximalMu:          float64Ptr(0.1),
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid fedprox scenario, got %v", err)
	}
}

func TestValidateScenarioRejectsScikitLearnFedProxStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "scikit_learn",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            proxAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
			ProximalMu:          float64Ptr(0.1),
		},
		ScikitLearn: &model.ScikitLearnFramework{
			Model: model.ScikitLearnModel{
				Kind:      "logistic_regression",
				NFeatures: 4,
				Classes:   []interface{}{0, 1},
			},
		},
	}

	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("expected validation error for scikit_learn fedprox scenario")
	}
}

func TestValidateScenarioAcceptsFedAdamStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "scikit_learn",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            adamAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
			Eta:                 float64Ptr(0.05),
			EtaL:                float64Ptr(0.2),
			Beta1:               float64Ptr(0.85),
			Beta2:               float64Ptr(0.95),
			Tau:                 float64Ptr(0.01),
		},
		ScikitLearn: &model.ScikitLearnFramework{
			Model: model.ScikitLearnModel{
				Kind:      "logistic_regression",
				NFeatures: 4,
				Classes:   []interface{}{0, 1},
			},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid fedadam scenario, got %v", err)
	}
}

func TestValidateScenarioAcceptsFedAdagradStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "scikit_learn",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            adagradAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
			Eta:                 float64Ptr(0.05),
			EtaL:                float64Ptr(0.2),
			Tau:                 float64Ptr(0.01),
		},
		ScikitLearn: &model.ScikitLearnFramework{
			Model: model.ScikitLearnModel{
				Kind:      "logistic_regression",
				NFeatures: 4,
				Classes:   []interface{}{0, 1},
			},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid fedadagrad scenario, got %v", err)
	}
}

func TestValidateScenarioAcceptsFedYogiStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     defaultFramework,
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            yogiAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
			Eta:                 float64Ptr(0.02),
			EtaL:                float64Ptr(0.05),
			Beta1:               float64Ptr(0.85),
			Beta2:               float64Ptr(0.95),
			Tau:                 float64Ptr(0.01),
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid fedyogi scenario, got %v", err)
	}
}

func TestValidateScenarioRejectsInvalidFedAdamBeta(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     defaultFramework,
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            adamAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 2,
			Beta1:               float64Ptr(1),
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("expected validation error for invalid fedadam beta_1")
	}
}

func TestValidateScenarioRejectsInsufficientBulyanClients(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     defaultFramework,
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            bulyanAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 6,
			NumMaliciousNodes:   1,
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("expected validation error for insufficient bulyan clients")
	}
}

func TestValidateScenarioAcceptsXGBoostBlock(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "xgboost",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation:   model.Aggregation{Strategy: defaultXGBoostAggregationStrategy, FractionFit: 1, FractionEvaluate: 1, MinAvailableClients: 1},
		XGBoost: &model.XGBoostFramework{
			Model: model.XGBoostModel{
				Kind:      "booster",
				NFeatures: 4,
			},
			Params: map[string]any{
				"objective": "binary:logistic",
			},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid xgboost scenario, got %v", err)
	}
}

func TestValidateScenarioAcceptsXGBoostCyclicStrategy(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Framework:     "xgboost",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation:   model.Aggregation{Strategy: cyclicXGBoostAggregationStrategy, FractionFit: 1, FractionEvaluate: 1, MinAvailableClients: 1},
		XGBoost: &model.XGBoostFramework{
			Model: model.XGBoostModel{
				Kind:      "booster",
				NFeatures: 4,
			},
			Params: map[string]any{
				"objective": "binary:logistic",
			},
		},
	}

	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatalf("expected valid xgboost cyclic scenario, got %v", err)
	}
}

func TestDecodeScenarioLogsDeprecationWarnings(t *testing.T) {
	var logs strings.Builder
	logger := slog.New(slog.NewTextHandler(&logs, nil))
	originalLogger := slog.Default()
	slog.SetDefault(logger)
	defer slog.SetDefault(originalLogger)

	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "model.json")
	if err := os.WriteFile(modelPath, []byte(`{"class_name":"Sequential","config":{"layers":[]}}`), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	service := &serverLogicServiceImpl{}
	body := io.NopCloser(strings.NewReader(`{
		"name":"demo",
		"framework":"tensorflow",
		"dataset":"dataset",
		"label_column":"label",
		"preprocessing":{"drop_columns":[],"drop_rows":0,"shuffle":false,"split":{"train":0.7,"eval":0.2,"test":0.1}},
		"hyper":{"server_rounds":1,"local_epochs":1,"batch_size":16},
		"strategy":{"fraction_fit":1,"fraction_evaluate":1,"min_available_clients":1},
		"model_file":"` + modelPath + `"
	}`))

	if _, err := service.DecodeScenario(body); err != nil {
		t.Fatalf("DecodeScenario returned error: %v", err)
	}

	output := logs.String()
	for _, field := range []string{"hyper", "strategy", "model_file"} {
		if !strings.Contains(output, "field="+field) {
			t.Fatalf("expected deprecation warning for %s, got %q", field, output)
		}
	}
}

func TestScenarioForClientKeepsCurrentShape(t *testing.T) {
	service := &serverLogicServiceImpl{}
	scenario := model.Scenario{
		Name:        "demo",
		Framework:   defaultFramework,
		Datasets:    map[string]string{"client-a": "dataset-a", "client-b": "dataset-b"},
		LabelColumn: "label",
		Preprocessing: model.Preprocessing{
			DropColumns: []string{},
			Split:       model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1},
		},
		Training: model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            defaultAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 1,
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	clientScenario := service.scenarioForClient(scenario, "client-a")

	payload, err := json.Marshal(clientScenario)
	if err != nil {
		t.Fatalf("marshal scenario: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal scenario: %v", err)
	}

	if decoded["dataset"] != "dataset-a" {
		t.Fatalf("expected client dataset to be narrowed, got %#v", decoded["dataset"])
	}
	for _, field := range []string{"datasets", "hyper", "strategy", "model", "model_file"} {
		if _, ok := decoded[field]; ok {
			t.Fatalf("expected current scenario shape without top-level %q, got %#v", field, decoded[field])
		}
	}
}

func TestSetFlowerConfigWritesCurrentScenarioShape(t *testing.T) {
	service := &serverLogicServiceImpl{}
	originalShareDir := shareDir
	shareDir = t.TempDir()
	defer func() {
		shareDir = originalShareDir
	}()

	scenario := &model.Scenario{
		Name:        "demo",
		Framework:   defaultFramework,
		Dataset:     "dataset-a",
		LabelColumn: "label",
		Preprocessing: model.Preprocessing{
			DropColumns: []string{},
			Split:       model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1},
		},
		Training: model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation: model.Aggregation{
			Strategy:            defaultAggregationStrategy,
			FractionFit:         1,
			FractionEvaluate:    1,
			MinAvailableClients: 1,
		},
		TensorFlow: &model.TensorFlowFramework{
			Model: model.TensorFlowModel{Config: map[string]interface{}{"class_name": "Sequential"}},
		},
	}

	service.SetFlowerConfig(scenario)

	cfg, err := ini.Load(filepath.Join(shareDir, "config.ini"))
	if err != nil {
		t.Fatalf("load config.ini: %v", err)
	}

	if got := cfg.Section("serverapp").Key("changed").String(); got != "1" {
		t.Fatalf("expected changed marker to be 1, got %q", got)
	}

	var decoded map[string]any
	if err := json.Unmarshal([]byte(cfg.Section("serverapp").Key("scenario").String()), &decoded); err != nil {
		t.Fatalf("decode stored scenario: %v", err)
	}

	if decoded["framework"] != defaultFramework {
		t.Fatalf("expected stored framework %q, got %#v", defaultFramework, decoded["framework"])
	}
	for _, field := range []string{"hyper", "strategy", "model", "model_file"} {
		if _, ok := decoded[field]; ok {
			t.Fatalf("expected stored scenario without top-level %q, got %#v", field, decoded[field])
		}
	}
}

func TestValidatePyTorchScenario(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})
	scenario := &model.Scenario{
		Name: "torch", Framework: "pytorch", Dataset: "sensor", LabelColumn: "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: .7, Eval: .2, Test: .1}},
		Training:      model.Training{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Aggregation:   model.Aggregation{Strategy: "fedavg", FractionFit: 1, FractionEvaluate: 1, MinAvailableClients: 2},
		PyTorch:       &model.PyTorchFramework{Model: model.PyTorchModel{Kind: "mlp", NFeatures: 12, HiddenSizes: []int{16}}},
	}
	if err := service.ValidateScenario(scenario); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(scenario)
	if err != nil {
		t.Fatal(err)
	}
	var restored model.Scenario
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.PyTorch == nil || restored.PyTorch.Model.HiddenSizes[0] != 16 {
		t.Fatal("lost PyTorch configuration")
	}
	for _, strategy := range []string{"fedprox", "fedxgbbagging"} {
		scenario.Aggregation.Strategy = strategy
		if err := service.ValidateScenario(scenario); err == nil {
			t.Fatalf("accepted %s", strategy)
		}
	}
	scenario.Aggregation.Strategy = "fedavg"
	scenario.Aggregation.Privacy = &model.PrivacyConfig{Mode: "central_dp"}
	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("accepted untested privacy mode")
	}
	scenario.Aggregation.Privacy = nil
	scenario.PyTorch.Model.HiddenSizes = nil
	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("accepted MLP without hidden layers")
	}
}
