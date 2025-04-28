package service

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
)

type KafkaService interface {
	PublishPrepareScenario(scenario model.Scenario)
	ConsumePreparedScenario(ctx context.Context) model.Scenario
}
