package impl

import (
	"github.com/idlab-discover/AIDE-FL/client/orchestrator/service"
)

type orchestratorServiceImpl struct {
}

func NewOrchestratorServiceImpl() service.OrchestratorService {
	return &orchestratorServiceImpl{}
}
