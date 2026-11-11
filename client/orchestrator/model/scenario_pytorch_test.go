package model

import (
	"encoding/json"
	"testing"
)

func TestPyTorchConfigurationSurvivesForwarding(t *testing.T) {
	input := []byte(`{"framework":"pytorch","pytorch":{"model":{"kind":"mlp","n_features":12,"hidden_sizes":[32,16],"seed":42},"optimizer":{"kind":"adam","learning_rate":0.001}}}`)
	var scenario Scenario
	if err := json.Unmarshal(input, &scenario); err != nil {
		t.Fatal(err)
	}
	output, err := json.Marshal(scenario)
	if err != nil {
		t.Fatal(err)
	}
	var restored Scenario
	if err := json.Unmarshal(output, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.Framework != "pytorch" || restored.PyTorch == nil {
		t.Fatal("lost PyTorch framework")
	}
	model := restored.PyTorch["model"].(map[string]interface{})
	if model["n_features"] != float64(12) || model["seed"] != float64(42) {
		t.Fatal("lost model settings")
	}
	optimizer := restored.PyTorch["optimizer"].(map[string]interface{})
	if optimizer["learning_rate"] != 0.001 {
		t.Fatal("lost optimizer settings")
	}
}
