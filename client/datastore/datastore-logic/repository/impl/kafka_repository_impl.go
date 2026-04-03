package impl

import (
	"context"
	"log/slog"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/repository"
	"github.com/segmentio/kafka-go"
)

type kafkaRepositoryImpl struct {
	prepareReader        *kafka.Reader
	preparedWriter       *kafka.Writer
	generalMessageWriter *kafka.Writer
}

func NewKafkaRepositoryImpl(prepareReader *kafka.Reader, preparedWriter *kafka.Writer, generalMessageWriter *kafka.Writer) repository.KafkaRepository {
	return &kafkaRepositoryImpl{
		prepareReader:        prepareReader,
		preparedWriter:       preparedWriter,
		generalMessageWriter: generalMessageWriter,
	}
}

func (repo *kafkaRepositoryImpl) ReadPrepareScenario(ctx context.Context) kafka.Message {
	m, err := repo.prepareReader.FetchMessage(ctx)
	exception.PanicLogging(err, "Unable to read prepare scenario message")

	return m
}

func (repo *kafkaRepositoryImpl) WritePreparedScenario(msg kafka.Message) {
	err := repo.preparedWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to write prepared scenario message")
}

func (repo *kafkaRepositoryImpl) WriteGeneralMessage(msg kafka.Message) {
	err := repo.generalMessageWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to write general message")
}

func (repo *kafkaRepositoryImpl) CommitPrepareScenario(ctx context.Context, msg kafka.Message) {
	slog.Info("Commiting prepare scenario message")

	err := repo.prepareReader.CommitMessages(ctx, msg)
	exception.PanicLogging(err, "Unable to commit prepare scenario", "message", msg)
}

func (repo *kafkaRepositoryImpl) Close() {
	defer repo.prepareReader.Close()
	defer repo.preparedWriter.Close()
	defer repo.generalMessageWriter.Close()
}
