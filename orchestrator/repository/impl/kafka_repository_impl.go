package impl

import (
	"context"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
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

func (repo *kafkaRepositoryImpl) WritePrepareScenario(msg kafka.Message) {
	err := repo.prepareWriter.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to publish prepare scenario")
}

func (repo *kafkaRepositoryImpl) ReadPreparedScenario(ctx context.Context) kafka.Message {
	m, err := repo.preparedReader.ReadMessage(ctx)
	exception.PanicLogging(err, "Unable to read prepared scenario message")

	return m
}

func (repo *kafkaRepositoryImpl) CommitPreparedScenario(ctx context.Context, msg kafka.Message) {
	slog.Info("Commiting prepare scenario message")

	err := repo.preparedReader.CommitMessages(ctx, msg)
	exception.PanicLogging(err, "Unable to commit prepare scenario", "message", msg)
}

func (repo *kafkaRepositoryImpl) Close() {
	defer repo.prepareWriter.Close()
	defer repo.preparedReader.Close()
}
