package repository

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
)

type KafkaRepository interface {
	ConsumePrepareScenario(ctx context.Context, handler func(model.Scenario))
	PublishPreparedScenario(scenario model.Scenario)
	Close()
}
