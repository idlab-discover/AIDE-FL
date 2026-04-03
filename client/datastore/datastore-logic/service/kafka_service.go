package service

import (
	"context"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/model"
)

type KafkaService interface {
	ConsumePrepareScenario(ctx context.Context, handler func(model.Scenario) error)
	PublishPreparedScenario(scenario model.Scenario)
	PublishGeneralMessage(message model.GeneralMessage)
}
