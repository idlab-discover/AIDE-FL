package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service"
)

type kafkaServiceImpl struct {
	repository.KafkaPublishRepository
}

func NewKafkaServiceImpl(kafkaRepo *repository.KafkaPublishRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaPublishRepository: *kafkaRepo}
}

func (service *kafkaServiceImpl) PublishScenarioStatus(scenario model.Scenario) {
	service.KafkaPublishRepository.PublishScenarioStatus(scenario)
}
