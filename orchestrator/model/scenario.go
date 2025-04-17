package model

type Scenario struct {
	Name          string        `json:"name"`
	Dataset       string        `json:"dataset"`
	Preprocessing Preprocessing `json:"preprocessing"`
	Model         []Layer       `json:"model"`
}
