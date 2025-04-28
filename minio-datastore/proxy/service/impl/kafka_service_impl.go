package impl

import (
	"context"

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

func (service *kafkaServiceImpl) ConsumePrepareScenario(ctx context.Context, handler func(model.Scenario)) {
	service.KafkaRepository.ConsumePrepareScenario(ctx, handler)
}

func (service *kafkaServiceImpl) PublishPreparedScenario(scenario model.Scenario) {
	service.KafkaRepository.PublishPreparedScenario(scenario)
}
