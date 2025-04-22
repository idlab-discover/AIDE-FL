package repository

import "github.com/MartienskieNum1/AIDE-FL/orchestrator/model"

type MinioProxyRepository interface {
	SendScenario(scenario model.Scenario) model.Scenario
}
