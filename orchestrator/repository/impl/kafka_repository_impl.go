package impl

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/repository"
	"github.com/segmentio/kafka-go"
)

type kafkaRepository struct {
	prepareScenarioWriter *kafka.Writer
}

func NewKafkaRepositoryImpl(kafkaPrepareScenarioWriter *kafka.Writer) repository.KafkaRepository {
	return &kafkaRepository{
		prepareScenarioWriter: kafkaPrepareScenarioWriter,
	}
}

func (repo *kafkaRepository) PublishPrepareScenario(scenario model.Scenario) {
	slog.Info("Publishing prepare", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Name),
		Value: byteArray,
	}

	err = repo.prepareScenarioWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to publish prepare", "scenario", scenario.Name)
}

func (repo *kafkaRepository) Close() {
	err := repo.prepareScenarioWriter.Close()
	exception.PanicLogging(err, "Unable to close prepare scenario writer")
}
