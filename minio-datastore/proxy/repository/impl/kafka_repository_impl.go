package impl

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
	"github.com/segmentio/kafka-go"
)

type kafkaRepositoryImpl struct {
	prepareReader  *kafka.Reader
	preparedWriter *kafka.Writer
}

func NewKafkaPublishRepositoryImpl(prepareReader *kafka.Reader, preparedWriter *kafka.Writer) repository.KafkaRepository {
	return &kafkaRepositoryImpl{
		prepareReader:  prepareReader,
		preparedWriter: preparedWriter,
	}
}

func (repo *kafkaRepositoryImpl) ConsumePrepareScenario(ctx context.Context, handler func(model.Scenario)) {
	slog.Info("Consuming prepare scenario messages")

	for {
		m, err := repo.prepareReader.ReadMessage(ctx)
		exception.PanicLogging(err, "Unable to read prepare scenario message")

		var scenario model.Scenario
		err = json.Unmarshal(m.Value, &scenario)
		exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

		handler(scenario)
		repo.CommitPrepareScenario(ctx, m)
	}
}

func (repo *kafkaRepositoryImpl) CommitPrepareScenario(ctx context.Context, msgs ...kafka.Message) {
	slog.Info("Commiting prepare scenario message")

	err := repo.prepareReader.CommitMessages(ctx, msgs...)
	exception.PanicLogging(err, "Unable to commit prepare scenario", "message", msgs)
}

func (repo *kafkaRepositoryImpl) PublishPreparedScenario(scenario model.Scenario) {
	slog.Info("Publishing prepared", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Name),
		Value: byteArray,
	}

	err = repo.preparedWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to publish prepared", "scenario", scenario.Name)
}

func (repo *kafkaRepositoryImpl) Close() {
	defer repo.prepareReader.Close()
	defer repo.preparedWriter.Close()
}
