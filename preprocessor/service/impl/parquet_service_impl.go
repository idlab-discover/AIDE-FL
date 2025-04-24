package impl

import (
	"bytes"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/custom"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/exception"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
	"github.com/xitongsys/parquet-go/reader"
)

type parquetServiceImpl struct {
}

func NewParquetService() service.ParquetService {
	return &parquetServiceImpl{}
}

func (service *parquetServiceImpl) ExtractRows(buf *bytes.Buffer) []interface{} {
	memFile := custom.NewInMemoryFile(buf.Bytes())

	pr, err := reader.NewParquetReader(memFile, nil, 4)
	exception.PanicLogging(err, "Unable to create Parquet reader")
	defer pr.ReadStop()

	n := int(pr.GetNumRows())
	rows := make([]interface{}, n)
	err = pr.Read(&rows)
	exception.PanicLogging(err, "Unable to read records")

	return rows
}
