package model

type Scenario struct {
	Name          string                 `json:"name"`
	Dataset       string                 `json:"dataset"`
	LabelColumn   string                 `json:"label_column"`
	Preprocessing map[string]interface{} `json:"preprocessing"`
	Hyper         map[string]interface{} `json:"hyper"`
	Strategy      map[string]interface{} `json:"strategy"`
	ModelConfig   map[string]interface{} `json:"model_config"`
	Model         map[string]interface{} `json:"model"`
}
