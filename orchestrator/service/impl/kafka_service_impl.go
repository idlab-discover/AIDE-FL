package impl

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/repository"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/service"
	"github.com/segmentio/kafka-go"
)

type kafkaServiceImpl struct {
	repository.KafkaRepository
}

func NewKafkaServiceImpl(kafkaRepository *repository.KafkaRepository) service.KafkaService {
	return &kafkaServiceImpl{KafkaRepository: *kafkaRepository}
}

func (service *kafkaServiceImpl) PublishPrepareScenario(scenario model.Scenario) {
	slog.Info("Publishing prepare", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Name),
		Value: byteArray,
	}

	service.KafkaRepository.WritePrepareScenario(msg)
}

func (service *kafkaServiceImpl) PublishPreprocessScenario(scenario model.Scenario) {
	slog.Info("Publishing preprocess", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Name),
		Value: byteArray,
	}

	service.KafkaRepository.WritePreprocessScenario(msg)
}

func (service *kafkaServiceImpl) ConsumePreparedScenario(ctx context.Context) model.Scenario {
	slog.Info("Consuming prepared scenario messages")

	m := service.KafkaRepository.ReadPreparedScenario(ctx)

	var scenario model.Scenario
	err := json.Unmarshal(m.Value, &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	slog.Info("Received prepared", "scenario", scenario.Name)

	service.KafkaRepository.CommitPreparedScenario(ctx, m)

	return scenario
}

func (service *kafkaServiceImpl) ConsumePreprocessedScenario(ctx context.Context) model.Scenario {
	slog.Info("Consuming preprocessed scenario messages")

	m := service.KafkaRepository.ReadPreprocessedScenario(ctx)

	var scenario model.Scenario
	err := json.Unmarshal(m.Value, &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	slog.Info("Received prepared", "scenario", scenario.Name)

	service.KafkaRepository.CommitPreprocessedScenario(ctx, m)

	return scenario
}
