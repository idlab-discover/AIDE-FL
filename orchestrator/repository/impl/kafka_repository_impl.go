package impl

import (
	"context"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/orchestrator/repository"
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

func (repo *kafkaReaderRepositoryImpl) Read(ctx context.Context) kafka.Message {
	m, err := repo.Reader.ReadMessage(ctx)
	exception.PanicLogging(err, "Unable to read Kafka message")

	return m
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
	defer repo.Reader.Close()
}

func (repo *kafkaWriterRepositoryImpl) Close() {
	defer repo.Writer.Close()
}
