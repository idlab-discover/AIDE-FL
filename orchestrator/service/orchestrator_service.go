package service

import "github.com/MartienskieNum1/AIDE-FL/orchestrator/model"

type OrchestratorService interface {
	IsPreprocessed(scenario model.Scenario) bool
}
