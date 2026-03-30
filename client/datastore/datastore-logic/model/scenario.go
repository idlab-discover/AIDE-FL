package model

type Scenario struct {
	PrepID        string                 `json:"prep_id,omitempty"`
	Name          string                 `json:"name"`
	Dataset       string                 `json:"dataset"`
	LabelColumn   string                 `json:"label_column"`
	Preprocessing map[string]interface{} `json:"preprocessing"`
	Hyper         map[string]interface{} `json:"hyper"`
	Strategy      map[string]interface{} `json:"strategy"`
	ModelFile     string                 `json:"model_file,omitempty"`
	Model         map[string]interface{} `json:"model"`
}
