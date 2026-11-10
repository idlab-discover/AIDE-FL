package model

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestScenarioPreservesPrivacyConfiguration(t *testing.T) {
	input := []byte(`{"name":"private-sensors","aggregation":{"strategy":"FedAvg","privacy":{"mode":"central_dp","noise_multiplier":1.2,"clipping_norm":0.5,"num_sampled_clients":2,"population_size":3,"target_delta":0.00001,"max_epsilon":8}}}`)
	var scenario Scenario
	if err := json.Unmarshal(input, &scenario); err != nil {
		t.Fatal(err)
	}
	output, err := json.Marshal(scenario)
	if err != nil {
		t.Fatal(err)
	}
	var before, after map[string]any
	if err := json.Unmarshal(input, &before); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(output, &after); err != nil {
		t.Fatal(err)
	}
	want := before["aggregation"].(map[string]any)["privacy"]
	got := after["aggregation"].(map[string]any)["privacy"]
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("privacy settings changed in transit: got %v, want %v", got, want)
	}
}

func TestScenarioOmitsPrivacyWhenUnconfigured(t *testing.T) {
	output, err := json.Marshal(Scenario{})
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(output, &decoded); err != nil {
		t.Fatal(err)
	}
	if _, ok := decoded["aggregation"].(map[string]any)["privacy"]; ok {
		t.Fatal("unconfigured privacy must not become an enabled mechanism")
	}
}

func TestScenarioRejectsMisspelledPrivacyBudget(t *testing.T) {
	var scenario Scenario
	err := json.Unmarshal([]byte(`{"aggregation":{"privacy":{"mode":"central_dp","max_epslion":8}}}`), &scenario)
	if err == nil {
		t.Fatal("a misspelled privacy budget must not silently become an unlimited budget")
	}
}
