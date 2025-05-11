package repository

import (
	"context"

	"github.com/segmentio/kafka-go"
)

type KafkaRepository interface {
	WritePrepareScenario(msg kafka.Message)
	WritePreprocessScenario(msg kafka.Message)
	ReadPreparedScenario(ctx context.Context) kafka.Message
	ReadPreprocessedScenario(ctx context.Context) kafka.Message
	CommitPreparedScenario(ctx context.Context, msg kafka.Message)
	CommitPreprocessedScenario(ctx context.Context, msg kafka.Message)
	Close()
}
