package model

import (
	"bytes"
	"encoding/json"
)

type Scenario struct {
	Name               string                `json:"name" validate:"required"`
	Framework          string                `json:"framework" validate:"required"`
	Dataset            string                `json:"dataset,omitempty"`
	DatasetFile        string                `json:"dataset_file,omitempty"`
	Datasets           map[string]string     `json:"datasets,omitempty"`
	HoldOutDataset     string                `json:"hold_out_dataset"`
	HoldOutDatasetFile string                `json:"hold_out_dataset_file,omitempty"`
	Centralised        bool                  `json:"centralised"`
	LabelColumn        string                `json:"label_column" validate:"required"`
	Preprocessing      Preprocessing         `json:"preprocessing" validate:"required"`
	Training           Training              `json:"training" validate:"required"`
	Aggregation        Aggregation           `json:"aggregation" validate:"required"`
	TensorFlow         *TensorFlowFramework  `json:"tensorflow,omitempty"`
	ScikitLearn        *ScikitLearnFramework `json:"scikit_learn,omitempty"`
	PyTorch            *PyTorchFramework     `json:"pytorch,omitempty"`
	XGBoost            *XGBoostFramework     `json:"xgboost,omitempty"`

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
	Privacy             *PrivacyConfig `json:"privacy,omitempty"`
	Strategy            string         `json:"strategy" validate:"required"`
	FractionFit         float64        `json:"fraction_fit" validate:"required,gt=0,lte=1.0"`
	FractionEvaluate    float64        `json:"fraction_evaluate" validate:"required,gt=0,lte=1.0"`
	MinAvailableClients int            `json:"min_available_clients" validate:"required,gte=1"`
	NumMaliciousNodes   int            `json:"num_malicious_nodes,omitempty" validate:"gte=0"`
	ProximalMu          *float64       `json:"proximal_mu,omitempty" validate:"omitempty,gte=0"`
	Eta                 *float64       `json:"eta,omitempty" validate:"omitempty,gt=0"`
	EtaL                *float64       `json:"eta_l,omitempty" validate:"omitempty,gt=0"`
	Beta1               *float64       `json:"beta_1,omitempty" validate:"omitempty,gt=0,lt=1"`
	Beta2               *float64       `json:"beta_2,omitempty" validate:"omitempty,gt=0,lt=1"`
	Tau                 *float64       `json:"tau,omitempty" validate:"omitempty,gt=0"`
}

// PrivacyConfig configures the client-level central DP mechanism in Flower.
// The Flower boundary validates mechanism and framework compatibility.
type PrivacyConfig struct {
	Mode              string   `json:"mode"`
	NoiseMultiplier   float64  `json:"noise_multiplier"`
	ClippingNorm      float64  `json:"clipping_norm"`
	NumSampledClients int      `json:"num_sampled_clients"`
	PopulationSize    int      `json:"population_size"`
	TargetDelta       float64  `json:"target_delta"`
	MaxEpsilon        *float64 `json:"max_epsilon,omitempty"`
}

// UnmarshalJSON rejects misspelled privacy options rather than silently dropping them.
func (p *PrivacyConfig) UnmarshalJSON(data []byte) error {
	type config PrivacyConfig
	var decoded config
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	*p = PrivacyConfig(decoded)
	return nil
}

type LegacyAggregation struct {
	FractionFit         float64 `json:"fraction_fit" validate:"required,gt=0,lte=1.0"`
	FractionEvaluate    float64 `json:"fraction_evaluate" validate:"required,gt=0,lte=1.0"`
	MinAvailableClients int     `json:"min_available_clients" validate:"required,gte=1"`
}

// PyTorchFramework describes the supported binary tabular classifiers.
type PyTorchFramework struct {
	Model     PyTorchModel      `json:"model" validate:"required"`
	Optimizer *PyTorchOptimizer `json:"optimizer,omitempty"`
}

type PyTorchModel struct {
	Kind         string    `json:"kind" validate:"oneof=logistic_regression mlp"`
	NFeatures    int       `json:"n_features" validate:"gte=1"`
	HiddenSizes  []int     `json:"hidden_sizes,omitempty" validate:"dive,gt=0"`
	Seed         *int64    `json:"seed,omitempty" validate:"omitempty,gte=0,lt=4294967296"`
	FeatureMean  []float64 `json:"feature_mean,omitempty"`
	FeatureScale []float64 `json:"feature_scale,omitempty"`
}

type PyTorchOptimizer struct {
	Kind         string   `json:"kind,omitempty" validate:"omitempty,oneof=adam sgd"`
	LearningRate *float64 `json:"learning_rate,omitempty" validate:"omitempty,gt=0"`
	WeightDecay  *float64 `json:"weight_decay,omitempty" validate:"omitempty,gte=0"`
}

// UnmarshalJSON prevents unsupported model and optimizer settings being discarded.
func (p *PyTorchFramework) UnmarshalJSON(data []byte) error {
	type config PyTorchFramework
	var decoded config
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	*p = PyTorchFramework(decoded)
	return nil
}
