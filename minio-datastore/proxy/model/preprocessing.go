package model

type Preprocessing struct {
	ExcludeColumns []string `json:"exclude_columns"`
	TrainTestSplit float64  `json:"train_test_split"`
	Scaling        string   `json:"scaling"`
	Normalize      bool     `json:"normalize"`
}
