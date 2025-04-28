package repository

import (
	"context"

	"github.com/segmentio/kafka-go"
)

type KafkaRepository interface {
	WritePrepareScenario(msg kafka.Message)
	ReadPreparedScenario(ctx context.Context) kafka.Message
	CommitPreparedScenario(ctx context.Context, msg kafka.Message)
	Close()
}
