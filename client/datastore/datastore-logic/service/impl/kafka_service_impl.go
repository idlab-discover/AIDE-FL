package impl

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/repository"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/service"
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

		exception.SetCommitter(service.KafkaRepository.CommitPrepareScenario, ctx, m) // So that Kafka message may be committed on error
		exception.SetPrepID(scenario.PrepID)
		func() {
			defer exception.ClearPrepID()
			defer exception.UnsetCommitter()
			handler(scenario)
		}()

		service.KafkaRepository.CommitPrepareScenario(ctx, m)
	}
}

func (service *kafkaServiceImpl) PublishPreparedScenario(scenario model.Scenario) {
	slog.Info("Publishing prepared", "scenario", scenario.Name, "prepId", scenario.PrepID)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.PrepID),
		Value: byteArray,
	}

	service.KafkaRepository.WritePreparedScenario(msg)
}

func (service *kafkaServiceImpl) PublishGeneralMessage(m model.GeneralMessage) {
	slog.Info("Publishing general message", "type", m.Type, "prepId", m.PrepID)

	byteArray, err := json.Marshal(m)
	exception.PanicLogging(err, "Unable to marshal general message to JSON")

	msg := kafka.Message{
		Key:   []byte(messageKey(m.PrepID, m.Type)),
		Value: byteArray,
	}

	service.KafkaRepository.WriteGeneralMessage(msg)
}

func messageKey(prepID string, fallback string) string {
	if prepID != "" {
		return prepID
	}

	return fallback
}
