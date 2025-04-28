package repository

import (
	"context"

	"github.com/segmentio/kafka-go"
)

type KafkaRepository interface {
	ReadPrepareScenario(ctx context.Context) kafka.Message
	CommitPrepareScenario(ctx context.Context, msg kafka.Message)
	WritePreparedScenario(msg kafka.Message)
	Close()
}
