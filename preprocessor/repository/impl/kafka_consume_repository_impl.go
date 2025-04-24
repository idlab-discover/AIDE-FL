package impl

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/configuration"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/exception"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/model"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/repository"
	"github.com/segmentio/kafka-go"
)

type kafkaConsumeRepository struct {
	*kafka.Reader
}

func NewKafkaConsumeRepositoryImpl(config configuration.Config) repository.KafkaConsumeRepository {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  []string{config.Get("KAFKA_ADDRESS")},
		Topic:    config.Get("KAFKA_TOPIC"),
		GroupID:  config.Get("KAFKA_GROUP"),
		MinBytes: 1e3,  // 1KB
		MaxBytes: 10e6, // 10MB
		MaxWait:  1 * time.Second,
		// Partition: 0, // or using GroupId
	})

	return &kafkaConsumeRepository{Reader: reader}
}

func (repo *kafkaConsumeRepository) ConsumeScenarioStatusEvents(ctx context.Context, handler func(model.Scenario)) {
	slog.Info("Consuming scenarios")

	for {
		m, err := repo.Reader.ReadMessage(ctx)
		exception.PanicLogging(err, "Unable to read Kafka message")

		var scenario model.Scenario
		err = json.Unmarshal(m.Value, &scenario)
		exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

		handler(scenario)
		repo.CommitMessages(ctx, m)
	}
}

func (repo *kafkaConsumeRepository) CommitMessages(ctx context.Context, msgs ...kafka.Message) {
	slog.Info("Commiting message")

	err := repo.Reader.CommitMessages(ctx, msgs...)
	exception.PanicLogging(err, "Unable to commit", "message", msgs)
}

func (repo *kafkaConsumeRepository) Close() {
	err := repo.Reader.Close()
	exception.PanicLogging(err, "Unable to close Kafka reader")
}
