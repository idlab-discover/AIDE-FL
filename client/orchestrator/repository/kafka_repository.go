package repository

import (
	"context"

	"github.com/segmentio/kafka-go"
)

type KafkaReaderRepository interface {
	Read(ctx context.Context) (kafka.Message, error)
	Commit(ctx context.Context, msg kafka.Message)
	Close()
}

type KafkaWriterRepository interface {
	Write(msg kafka.Message)
	Close()
}
