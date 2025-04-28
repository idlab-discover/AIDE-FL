package service

import "github.com/MartienskieNum1/AIDE-FL/orchestrator/model"

type KafkaService interface {
	PublishPrepareScenario(scenario model.Scenario)
}
