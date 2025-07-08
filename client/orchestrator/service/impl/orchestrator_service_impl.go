package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/service"
)

type orchestratorServiceImpl struct {
}

func NewOrchestratorServiceImpl() service.OrchestratorService {
	return &orchestratorServiceImpl{}
}
