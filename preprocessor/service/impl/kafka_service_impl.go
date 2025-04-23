package impl

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/repository"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type kafkaServiceImpl struct {
	repository.KafkaConsumeRepository
}

func NewKafkaServiceImpl(repo repository.KafkaConsumeRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaConsumeRepository: repo}
}

func (service *kafkaServiceImpl) ConsumeScenarioStatusEvents(ctx context.Context, handler func(model.Scenario)) {
	service.KafkaConsumeRepository.ConsumeScenarioStatusEvents(ctx, handler)
}
