package model

type Scenario struct {
	Name          string        `json:"name"`
	Dataset       string        `json:"dataset"`
	Preprocessing Preprocessing `json:"preprocessing"`
	Model         []Layer       `json:"model"`
	Status        string        `json:"status"`
}

type Preprocessing struct {
	DropColumns []string `json:"drop_columns"`
	Shuffle     bool     `json:"shuffle"`
	Split       Split    `json:"split"`
	Normalize   bool     `json:"normalize"`
}

type Layer struct {
	ClassName string                 `json:"class_name"`
	Config    map[string]interface{} `json:"config"`
}

type Split struct {
	Train      float32 `json:"train"`
	Validation float32 `json:"eval"`
	Test       float32 `json:"test"`
}
