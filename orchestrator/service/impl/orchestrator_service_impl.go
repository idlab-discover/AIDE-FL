package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/service"
)

type orchestratorServiceImpl struct {
}

func NewOrchestratorServiceImpl() service.OrchestratorService {
	return &orchestratorServiceImpl{}
}

func (service *orchestratorServiceImpl) IsPreprocessed(scenario model.Scenario) bool {
	return scenario.Status != "not preprocessed"
}
