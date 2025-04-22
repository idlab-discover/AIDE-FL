package impl

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
	"github.com/segmentio/kafka-go"
)

type kafkaRepository struct {
	*kafka.Writer
}

func NewKafkaRepositoryImpl(config configuration.Config) repository.KafkaRepository {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(config.Get("KAFKA_ADDRESS")),
		Topic:        config.Get("KAFKA_TOPIC"),
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
		BatchSize:    1,
		WriteTimeout: 5 * time.Second,
	}

	return &kafkaRepository{
		Writer: writer,
	}
}

func (repo *kafkaRepository) PublishDatasetStatus(scenario model.Scenario) {
	slog.Info("Publishing", "scenario", scenario.Name)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.Dataset),
		Value: byteArray,
	}

	err = repo.Writer.WriteMessages(context.Background(), msg)
	exception.PanicLogging(err, "Unable to publish", "scenario", scenario.Name)
}

func (repo *kafkaRepository) Close() {
	err := repo.Writer.Close()
	exception.PanicLogging(err, "Unable to close Kafka writer")
}
