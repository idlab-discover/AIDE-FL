package service

import "github.com/MartienskieNum1/AIDE-FL/preprocessor/model"

type PreprocessorService interface {
	SplitDataset(data [][]string, ratios model.SplitRatios) [][]string
	Shuffle(data *[][]string)
	Normalize(data *[][]string)
	DropColumns(data *[][]string)
}
