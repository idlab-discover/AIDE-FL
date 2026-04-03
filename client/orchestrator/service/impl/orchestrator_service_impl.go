package impl

import (
	"github.com/Str-Gen/AIDE-FL/client/orchestrator/service"
)

type orchestratorServiceImpl struct {
}

func NewOrchestratorServiceImpl() service.OrchestratorService {
	return &orchestratorServiceImpl{}
}
