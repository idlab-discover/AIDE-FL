package service

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
)

type KafkaService interface {
	ConsumeScenarioStatusEvents(ctx context.Context, handler func(model.Scenario))
}
