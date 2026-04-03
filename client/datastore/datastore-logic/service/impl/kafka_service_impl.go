package impl

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/repository"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/service"
	"github.com/segmentio/kafka-go"
)

type kafkaServiceImpl struct {
	repository.KafkaRepository
}

func NewKafkaServiceImpl(kafkaRepo *repository.KafkaRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaRepository: *kafkaRepo}
}

func (service *kafkaServiceImpl) ConsumePrepareScenario(ctx context.Context, handler func(model.Scenario) error) {
	slog.Info("Consuming prepare scenario messages")

	for {
		m := service.KafkaRepository.ReadPrepareScenario(ctx)
		service.handlePrepareScenario(ctx, m, handler)
	}
}

func (service *kafkaServiceImpl) handlePrepareScenario(ctx context.Context, m kafka.Message, handler func(model.Scenario) error) {
	var scenario model.Scenario
	err := json.Unmarshal(m.Value, &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	exception.SetPrepID(scenario.PrepID)
	defer exception.ClearPrepID()

	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(exception.LoggedError); !ok {
				service.PublishGeneralMessage(model.GeneralMessage{
					PrepID:  scenario.PrepID,
					Type:    "ERROR",
					Origin:  "datastore",
					Message: fmt.Sprintf("panic while preparing scenario: %v", r),
				})
			}

			service.KafkaRepository.CommitPrepareScenario(ctx, m)
		}
	}()

	if err := handler(scenario); err != nil {
		service.PublishGeneralMessage(model.GeneralMessage{
			PrepID:  scenario.PrepID,
			Type:    "ERROR",
			Origin:  "datastore",
			Message: fmt.Sprintf("unable to prepare scenario: %v", err),
		})
		service.KafkaRepository.CommitPrepareScenario(ctx, m)
		return
	}

	service.KafkaRepository.CommitPrepareScenario(ctx, m)
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
