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

type kafkaRepositoryImpl struct {
	prepareWriter  *kafka.Writer
	preparedReader *kafka.Reader
}

func NewKafkaRepositoryImpl(prepareWriter *kafka.Writer, preparedReader *kafka.Reader) repository.KafkaRepository {
	return &kafkaRepositoryImpl{
		prepareWriter:  prepareWriter,
		preparedReader: preparedReader,
	}
}

func (repo *kafkaRepositoryImpl) PublishPrepareScenario(scenario model.Scenario) {
	slog.Info("Publishing prepare", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Name),
		Value: byteArray,
	}

	err = repo.prepareWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to publish prepare", "scenario", scenario.Name)
}

func (repo *kafkaRepositoryImpl) ConsumePreparedScenario(ctx context.Context) model.Scenario {
	slog.Info("Consuming prepared scenario messages")

	m, err := repo.preparedReader.ReadMessage(ctx)
	exception.PanicLogging(err, "Unable to read prepared scenario message")

	var scenario model.Scenario
	err = json.Unmarshal(m.Value, &scenario)
	exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

	slog.Info("Received prepared", "scenario", scenario.Name)

	repo.CommitPreparedScenario(ctx, m)

	return scenario
}

func (repo *kafkaRepositoryImpl) CommitPreparedScenario(ctx context.Context, msgs ...kafka.Message) {
	slog.Info("Commiting prepare scenario message")

	err := repo.preparedReader.CommitMessages(ctx, msgs...)
	exception.PanicLogging(err, "Unable to commit prepare scenario", "message", msgs)
}

func (repo *kafkaRepositoryImpl) Close() {
	defer repo.prepareWriter.Close()
	defer repo.preparedReader.Close()
}
