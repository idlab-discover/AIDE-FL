package configuration

import (
	"time"

	"github.com/segmentio/kafka-go"
)

func NewKafkaPrepareReader(config Config) *kafka.Reader {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  []string{config.Get("KAFKA_ADDRESS")},
		Topic:    config.Get("KAFKA_PREPARE_TOPIC"),
		GroupID:  config.Get("KAFKA_GROUP"),
		MinBytes: 1,    // 1B
		MaxBytes: 10e6, // 10MB
		MaxWait:  1 * time.Second,
	})

	return reader
}

func NewKafkaPreparedWriter(config Config) *kafka.Writer {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(config.Get("KAFKA_ADDRESS")),
		Topic:        config.Get("KAFKA_PREPARED_TOPIC"),
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
		BatchSize:    1,
		WriteTimeout: 5 * time.Second,
	}

	return writer
}

func NewKafkaGeneralMessageWriter(config Config) *kafka.Writer {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(config.Get("KAFKA_ADDRESS")),
		Topic:        config.Get("KAFKA_MESSAGE_TOPIC"),
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
		BatchSize:    1,
		WriteTimeout: 5 * time.Second,
	}

	return writer
}
