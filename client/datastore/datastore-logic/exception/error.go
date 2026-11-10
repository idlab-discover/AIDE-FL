package exception

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/idlab-discover/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/idlab-discover/AIDE-FL/client/datastore/datastore-logic/service"
)

var (
	kafkaService           service.KafkaService
	currentPrepID          string
	publishingGeneralError bool
)

// LoggedError marks a panic that already emitted a datastore ERROR message.
type LoggedError struct {
	Err error
}

func (err LoggedError) Error() string {
	return err.Err.Error()
}

func RegisterKafkaService(service *service.KafkaService) {
	kafkaService = *service
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

		if kafkaService != nil && !publishingGeneralError {
			publishingGeneralError = true
			kafkaService.PublishGeneralMessage(model.GeneralMessage{
				PrepID:  currentPrepID,
				Type:    "ERROR",
				Origin:  "datastore",
				Message: formatMessage(msg, append(args, "error", err)...),
			})
			publishingGeneralError = false
		}

		panic(LoggedError{Err: err})
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
