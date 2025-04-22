package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service"
)

type kafkaServiceImpl struct {
	repository.KafkaRepository
}

func NewKafkaServiceImpl(kafkaRepo *repository.KafkaRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaRepository: *kafkaRepo}
}

func (service *kafkaServiceImpl) PublishDatasetStatus(scenario model.Scenario, status model.Status) {
	service.KafkaRepository.PublishDatasetStatus(scenario, status)
}

func (service *kafkaServiceImpl) Close() {
	service.KafkaRepository.Close()
}
