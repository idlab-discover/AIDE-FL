package model

type ScenarioMessage struct {
	PrepId   string   `json:"prep_id"` // uuid.UUID as string
	Scenario Scenario `json:"scenario"`
}
