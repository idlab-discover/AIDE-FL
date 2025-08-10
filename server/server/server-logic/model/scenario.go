package model

type Scenario struct {
	Name           string                 `json:"name" validate:"required"`
	Dataset        string                 `json:"dataset,omitempty"`
	Datasets       map[string]string      `json:"datasets,omitempty"`
	HoldOutDataset string                 `json:"hold_out_dataset"`
	Centralised    bool                   `json:"centralised"`
	LabelColumn    string                 `json:"label_column" validate:"required"`
	Preprocessing  Preprocessing          `json:"preprocessing" validate:"required"`
	Hyper          Hyper                  `json:"hyper" validate:"required"`
	Strategy       Strategy               `json:"strategy" validate:"required"`
	Model          map[string]interface{} `json:"model" validate:"required"`
}

type Preprocessing struct {
	DropColumns []string    `json:"drop_columns" validate:"required"`
	DropRows    int         `json:"drop_rows"`
	Shuffle     bool        `json:"shuffle"`
	Split       SplitConfig `json:"split" validate:"required"`
}

type SplitConfig struct {
	Train float64 `json:"train" validate:"required"`
	Eval  float64 `json:"eval" validate:"required"`
	Test  float64 `json:"test" validate:"required"`
}

type Hyper struct {
	ServerRounds int `json:"server_rounds" validate:"required"`
	LocalEpochs  int `json:"local_epochs" validate:"required"`
	BatchSize    int `json:"batch_size" validate:"required"`
}

type Strategy struct {
	FractionFit         float64 `json:"fraction_fit" validate:"required,gt=0,lte=1.0"`
	FractionEvaluate    float64 `json:"fraction_evaluate" validate:"required,gt=0,lte=1.0"`
	MinAvailableClients int     `json:"min_available_clients" validate:"required,gte=1"`
}
