package repository

import "github.com/MartienskieNum1/AIDE-FL/orchestrator/model"

type KafkaRepository interface {
	PublishPrepareScenario(scenario model.Scenario)
	Close()
}
