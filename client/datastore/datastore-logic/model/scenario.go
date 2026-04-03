package model

type Scenario struct {
	PrepID        string                 `json:"prep_id,omitempty"`
	Name          string                 `json:"name"`
	Framework     string                 `json:"framework"`
	Dataset       string                 `json:"dataset"`
	Centralised   bool                   `json:"centralised"`
	LabelColumn   string                 `json:"label_column"`
	Preprocessing map[string]interface{} `json:"preprocessing"`
	Training      map[string]interface{} `json:"training"`
	Aggregation   map[string]interface{} `json:"aggregation"`
	TensorFlow    map[string]interface{} `json:"tensorflow,omitempty"`
	ScikitLearn   map[string]interface{} `json:"scikit_learn,omitempty"`
	XGBoost       map[string]interface{} `json:"xgboost,omitempty"`
}
