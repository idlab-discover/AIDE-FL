package service

import "bytes"

type ParquetService interface {
	ExtractRows(buf *bytes.Buffer) []interface{}
}
