package impl

import (
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/service"
)

type orchestratorServiceImpl struct {
}

func NewOrchestratorServiceImpl() service.OrchestratorService {
	return &orchestratorServiceImpl{}
}

func (service *orchestratorServiceImpl) IsPreprocessed(scenario model.Scenario) bool {
	slog.Info("Scenario", "status", scenario.Status)
	return scenario.Status == "preprocessed"
}
