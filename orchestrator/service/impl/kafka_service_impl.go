package impl

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/repository"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/service"
)

type kafkaServiceImpl struct {
	repository.KafkaRepository
}

func NewKafkaServiceImpl(kafkaRepository *repository.KafkaRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaRepository: *kafkaRepository}
}

func (service *kafkaServiceImpl) PublishPrepareScenario(scenario model.Scenario) {
	service.KafkaRepository.PublishPrepareScenario(scenario)
}

func (service *kafkaServiceImpl) ConsumePreparedScenario(ctx context.Context) model.Scenario {
	return service.KafkaRepository.ConsumePreparedScenario(ctx)
}
