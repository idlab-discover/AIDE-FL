package model

type Scenario struct {
	Name           string                `json:"name" validate:"required"`
	Framework      string                `json:"framework" validate:"required"`
	Dataset        string                `json:"dataset,omitempty"`
	Datasets       map[string]string     `json:"datasets,omitempty"`
	HoldOutDataset string                `json:"hold_out_dataset"`
	Centralised    bool                  `json:"centralised"`
	LabelColumn    string                `json:"label_column" validate:"required"`
	Preprocessing  Preprocessing         `json:"preprocessing" validate:"required"`
	Training       Training              `json:"training" validate:"required"`
	Aggregation    Aggregation           `json:"aggregation" validate:"required"`
	TensorFlow     *TensorFlowFramework  `json:"tensorflow,omitempty"`
	ScikitLearn    *ScikitLearnFramework `json:"scikit_learn,omitempty"`
	XGBoost        *XGBoostFramework     `json:"xgboost,omitempty"`

	LegacyHyper     *Training              `json:"hyper,omitempty"`
	LegacyStrategy  *LegacyAggregation     `json:"strategy,omitempty"`
	LegacyModelFile string                 `json:"model_file,omitempty"`
	LegacyModel     map[string]interface{} `json:"model,omitempty"`
}

type TensorFlowFramework struct {
	Model TensorFlowModel `json:"model" validate:"required"`
}

type TensorFlowModel struct {
	ConfigFile string                 `json:"config_file,omitempty"`
	Config     map[string]interface{} `json:"config,omitempty"`
}

type ScikitLearnFramework struct {
	Model ScikitLearnModel `json:"model" validate:"required"`
}

type ScikitLearnModel struct {
	Kind       string        `json:"kind" validate:"required"`
	NFeatures  int           `json:"n_features" validate:"required,gte=1"`
	Classes    []interface{} `json:"classes" validate:"required,min=2"`
	Penalty    string        `json:"penalty,omitempty"`
	Solver     string        `json:"solver,omitempty"`
	MultiClass string        `json:"multi_class,omitempty"`
}

type XGBoostFramework struct {
	Model  XGBoostModel   `json:"model" validate:"required"`
	Params map[string]any `json:"params" validate:"required"`
}

type XGBoostModel struct {
	Kind      string `json:"kind" validate:"required"`
	NFeatures int    `json:"n_features" validate:"required,gte=1"`
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

type Training struct {
	ServerRounds int `json:"server_rounds" validate:"required"`
	LocalEpochs  int `json:"local_epochs" validate:"required"`
	BatchSize    int `json:"batch_size" validate:"required"`
}

type Aggregation struct {
	Strategy            string  `json:"strategy" validate:"required"`
	FractionFit         float64 `json:"fraction_fit" validate:"required,gt=0,lte=1.0"`
	FractionEvaluate    float64 `json:"fraction_evaluate" validate:"required,gt=0,lte=1.0"`
	MinAvailableClients int     `json:"min_available_clients" validate:"required,gte=1"`
}

type LegacyAggregation struct {
	FractionFit         float64 `json:"fraction_fit" validate:"required,gt=0,lte=1.0"`
	FractionEvaluate    float64 `json:"fraction_evaluate" validate:"required,gt=0,lte=1.0"`
	MinAvailableClients int     `json:"min_available_clients" validate:"required,gte=1"`
}
