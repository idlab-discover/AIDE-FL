package model

type Scenario struct {
	Name          string        `json:"name"`
	Dataset       string        `json:"dataset"`
	Preprocessing Preprocessing `json:"preprocessing"`
	Model         []Layer       `json:"model"`
}

type Preprocessing struct {
	ExcludeColumns []string    `json:"exclude_columns"`
	Shuffle        bool        `json:"shuffle"`
	SplitRatios    SplitRatios `json:"split_ratios"`
	Normalize      bool        `json:"normalize"`
}

type Layer struct {
	ClassName string                 `json:"class_name"`
	Config    map[string]interface{} `json:"config"`
}

type SplitRatios struct {
	Train      float32 `json:"train"`
	Validation float32 `json:"validation"`
	Test       float32 `json:"test"`
}
