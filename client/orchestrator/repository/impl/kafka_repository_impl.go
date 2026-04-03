package impl

import (
	"context"
	"log/slog"

	"github.com/Str-Gen/AIDE-FL/client/orchestrator/exception"
	"github.com/Str-Gen/AIDE-FL/client/orchestrator/repository"
	"github.com/segmentio/kafka-go"
)

type kafkaReaderRepositoryImpl struct {
	*kafka.Reader
}

type kafkaWriterRepositoryImpl struct {
	*kafka.Writer
}

func NewKafkaReaderRepositoryImpl(reader *kafka.Reader) repository.KafkaReaderRepository {
	return &kafkaReaderRepositoryImpl{Reader: reader}
}

func NewKafkaWriterRepositoryImpl(writer *kafka.Writer) repository.KafkaWriterRepository {
	return &kafkaWriterRepositoryImpl{Writer: writer}
}

func (repo *kafkaReaderRepositoryImpl) Read(ctx context.Context) (kafka.Message, error) {
	m, err := repo.Reader.FetchMessage(ctx)
	if err != nil {
		slog.Error("Unable to read Kafka message", "error", err)
		return kafka.Message{}, err
	}

	return m, err
}

func (repo *kafkaReaderRepositoryImpl) Commit(ctx context.Context, msg kafka.Message) {
	slog.Info("Commiting Kafka message")

	err := repo.Reader.CommitMessages(ctx, msg)
	exception.PanicLogging(err, "Unable to commit Kafka", "message", msg)
}

func (repo *kafkaWriterRepositoryImpl) Write(msg kafka.Message) {
	err := repo.Writer.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to write Kafka message")
}

func (repo *kafkaReaderRepositoryImpl) Close() {
	repo.Reader.Close()
}

func (repo *kafkaWriterRepositoryImpl) Close() {
	repo.Writer.Close()
}
