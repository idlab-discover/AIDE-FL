package repository

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
)

type KafkaConsumeRepository interface {
	ConsumeScenarioStatusEvents(ctx context.Context, handler func(model.Scenario))
	Close()
}
