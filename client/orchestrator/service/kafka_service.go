package service

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/model"
)

type KafkaService interface {
	PublishPrepareScenario(scenario model.Scenario)
	PublishPreprocessScenario(scenario model.Scenario)
	PublishTrainScenario(scenario model.Scenario)
	ConsumePreparedScenario(ctx context.Context, expectedPrepID string) (model.Scenario, error)
	ConsumePreprocessedScenario(ctx context.Context, expectedPrepID string) (model.Scenario, error)
	ConsumeTrainedScenario(ctx context.Context, expectedPrepID string) (model.Scenario, error)
	ConsumeGeneralMessage(ctx context.Context) (model.GeneralMessage, error)
}
