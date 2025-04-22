package service

import "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"

type KafkaService interface {
	PublishDatasetStatus(scenario model.Scenario, status model.Status)
	Close()
}
