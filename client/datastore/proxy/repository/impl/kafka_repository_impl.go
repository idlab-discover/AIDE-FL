package impl

import (
	"context"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/repository"
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

func (repo *kafkaRepositoryImpl) ReadPrepareScenario(ctx context.Context) kafka.Message {
	m, err := repo.prepareReader.ReadMessage(ctx)
	exception.PanicLogging(err, "Unable to read prepare scenario message")

	return m
}

func (repo *kafkaRepositoryImpl) WritePreparedScenario(msg kafka.Message) {
	err := repo.preparedWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to write prepared scenario message")
}

func (repo *kafkaRepositoryImpl) CommitPrepareScenario(ctx context.Context, msg kafka.Message) {
	slog.Info("Commiting prepare scenario message")

	err := repo.prepareReader.CommitMessages(ctx, msg)
	exception.PanicLogging(err, "Unable to commit prepare scenario", "message", msg)
}

func (repo *kafkaRepositoryImpl) Close() {
	defer repo.prepareReader.Close()
	defer repo.preparedWriter.Close()
}
