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
