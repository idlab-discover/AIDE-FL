package impl

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/model"
	"github.com/go-playground/validator/v10"
)

func TestDecodeScenarioResolvesModelFile(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "model.json")
	if err := os.WriteFile(modelPath, []byte(`{"class_name":"Sequential","config":{"layers":[]}}`), 0o644); err != nil {
		t.Fatalf("write model file: %v", err)
	}

	service := &serverLogicServiceImpl{}
	body := io.NopCloser(strings.NewReader(`{
		"name":"demo",
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
	if scenario.ModelFile != modelPath {
		t.Fatalf("expected model_file %q, got %q", modelPath, scenario.ModelFile)
	}
	if len(scenario.Model) == 0 {
		t.Fatal("expected model to be loaded from model_file")
	}
}

func TestValidateScenarioRequiresModelOrModelFile(t *testing.T) {
	service := &serverLogicServiceImpl{}
	service.Validate = validator.New()
	service.Validate.RegisterStructValidation(datasetsValidation, model.Scenario{})

	scenario := &model.Scenario{
		Name:          "demo",
		Dataset:       "dataset",
		LabelColumn:   "label",
		Preprocessing: model.Preprocessing{DropColumns: []string{}, Split: model.SplitConfig{Train: 0.7, Eval: 0.2, Test: 0.1}},
		Hyper:         model.Hyper{ServerRounds: 1, LocalEpochs: 1, BatchSize: 16},
		Strategy:      model.Strategy{FractionFit: 1, FractionEvaluate: 1, MinAvailableClients: 1},
	}

	if err := service.ValidateScenario(scenario); err == nil {
		t.Fatal("expected validation error when both model and model_file are missing")
	}
}
