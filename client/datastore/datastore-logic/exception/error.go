package exception

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/service"
	"github.com/segmentio/kafka-go"
)

var (
	kafkaService      service.KafkaService
	commitFunction    func(ctx context.Context, m kafka.Message)
	commitFunctionCtx context.Context
	commitFunctionMsg kafka.Message
	currentPrepID     string
)

func RegisterKafkaService(service *service.KafkaService) {
	kafkaService = *service
}

func SetCommitter(cb func(ctx context.Context, m kafka.Message), ctx context.Context, m kafka.Message) {
	commitFunction = cb
	commitFunctionCtx = ctx
	commitFunctionMsg = m
}

func UnsetCommitter() {
	commitFunction = nil
}

func SetPrepID(prepID string) {
	currentPrepID = prepID
}

func ClearPrepID() {
	currentPrepID = ""
}

func PanicLogging(err error, msg string, args ...any) {
	if err != nil {
		slog.Error(msg, append(args, "error", err)...)

		m := model.GeneralMessage{
			PrepID:  currentPrepID,
			Type:    "ERROR",
			Origin:  "datastore",
			Message: formatMessage(msg, append(args, "error", err)...),
		}

		kafkaService.PublishGeneralMessage(m)

		if commitFunction != nil {
			commitFunction(commitFunctionCtx, commitFunctionMsg)
		}

		panic(err)
	}
}

func ErrorLogging(err error, msg string, args ...any) {
	if err != nil {
		slog.Error(msg, append(args, "error", err)...)

		m := model.GeneralMessage{
			PrepID:  currentPrepID,
			Type:    "ERROR",
			Origin:  "datastore",
			Message: formatMessage(msg, append(args, "error", err)...),
		}

		kafkaService.PublishGeneralMessage(m)

		if commitFunction != nil {
			commitFunction(commitFunctionCtx, commitFunctionMsg)
		}
	}
}

func formatMessage(msg string, args ...any) string {
	var b strings.Builder
	b.WriteString(msg)

	for i := 0; i < len(args)-1; i += 2 {
		key, ok := args[i].(string)
		if !ok {
			continue // skip malformed pair
		}
		val := args[i+1]
		b.WriteString(fmt.Sprintf(" %s=%v", key, val))
	}

	return b.String()
}
