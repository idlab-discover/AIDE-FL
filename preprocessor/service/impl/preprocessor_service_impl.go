package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type preprocessorServiceImpl struct {
}

func NewPrepocessorServiceImpl() service.PreprocessorService {
	return &preprocessorServiceImpl{}
}

func (p *preprocessorServiceImpl) DropColumns(data *[][]string) {
	panic("unimplemented")
}

func (p *preprocessorServiceImpl) Normalize(data *[][]string) {
	panic("unimplemented")
}

func (p *preprocessorServiceImpl) Shuffle(data *[][]string) {
	panic("unimplemented")
}

func (p *preprocessorServiceImpl) SplitDataset(data [][]string, ratios model.SplitRatios) [][]string {
	panic("unimplemented")
}
