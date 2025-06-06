package service

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
)

type KafkaService interface {
	PublishPrepareScenario(scenario model.Scenario)
	PublishPreprocessScenario(scenario model.Scenario)
	PublishTrainScenario(scenario model.Scenario)
	ConsumePreparedScenario(ctx context.Context) model.Scenario
	ConsumePreprocessedScenario(ctx context.Context) model.Scenario
	ConsumeTrainedScenario(ctx context.Context) model.Scenario
}
