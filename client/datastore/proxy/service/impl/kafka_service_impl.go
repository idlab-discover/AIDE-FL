package impl

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/repository"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/service"
	"github.com/segmentio/kafka-go"
)

type kafkaServiceImpl struct {
	repository.KafkaRepository
}

func NewKafkaServiceImpl(kafkaRepo *repository.KafkaRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaRepository: *kafkaRepo}
}

func (service *kafkaServiceImpl) ConsumePrepareScenario(ctx context.Context, handler func(model.Scenario)) {
	slog.Info("Consuming prepare scenario messages")

	for {
		m := service.KafkaRepository.ReadPrepareScenario(ctx)

		var scenario model.Scenario
		err := json.Unmarshal(m.Value, &scenario)
		exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

		handler(scenario)
		service.KafkaRepository.CommitPrepareScenario(ctx, m)
	}
}

func (service *kafkaServiceImpl) PublishPreparedScenario(scenario model.Scenario) {
	slog.Info("Publishing prepared", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Name),
		Value: byteArray,
	}

	service.KafkaRepository.WritePreparedScenario(msg)
}
