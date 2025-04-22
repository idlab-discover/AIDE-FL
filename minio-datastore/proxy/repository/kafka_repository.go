package repository

import "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"

type KafkaRepository interface {
	PublishScenarioStatus(scenario model.Scenario)
	Close()
}
