package exception

import "log/slog"

func PanicLogging(err interface{}, msg string, args ...any) {
	if err != nil {
		slog.Error(msg, args...)
		panic(err)
	}
}
