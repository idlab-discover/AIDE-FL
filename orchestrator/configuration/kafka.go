package configuration

import (
	"time"

	"github.com/segmentio/kafka-go"
)

func NewKafkaPrepareWriter(config Config) *kafka.Writer {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(config.Get("KAFKA_ADDRESS")),
		Topic:        config.Get("KAFKA_PREPARE_TOPIC"),
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
		BatchSize:    1,
		WriteTimeout: 5 * time.Second,
	}

	return writer
}

func NewKafkaPreparedReader(config Config) *kafka.Reader {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers: []string{config.Get("KAFKA_ADDRESS")},
		Topic:   config.Get("KAFKA_PREPARED_TOPIC"),
		GroupID: config.Get("KAFKA_GROUP"),
		//MinBytes: 1e3,  // 1KB
		MaxBytes: 10e6, // 10MB
		//MaxWait:  1 * time.Second,
		// Partition: 0, // or using GroupId
	})

	return reader
}

func NewKafkaPreprocessWriter(config Config) *kafka.Writer {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(config.Get("KAFKA_ADDRESS")),
		Topic:        config.Get("KAFKA_PREPROCESS_TOPIC"),
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
		BatchSize:    1,
		WriteTimeout: 5 * time.Second,
	}

	return writer
}

func NewKafkaPreprocessedReader(config Config) *kafka.Reader {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  []string{config.Get("KAFKA_ADDRESS")},
		Topic:    config.Get("KAFKA_PREPROCESSED_TOPIC"),
		GroupID:  config.Get("KAFKA_GROUP"),
		MaxBytes: 10e6, // 10MB
	})

	return reader
}
