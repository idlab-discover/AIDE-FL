package repository

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
)

type KafkaRepository interface {
	PublishPrepareScenario(scenario model.Scenario)
	ConsumePreparedScenario(ctx context.Context) model.Scenario
	Close()
}
