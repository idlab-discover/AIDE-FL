package impl

import (
	"bytes"
	"path/filepath"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/exception"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/repository"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository) service.MinioService {
	return &minioServiceImpl{MinioRepository: *minioRepository}
}

// Loads dataset into memory
func (service *minioServiceImpl) LoadDataset(dataset string) *bytes.Buffer {
	obj := service.MinioRepository.GetObject(filepath.Join(dataset, dataset+".parquet"))
	defer obj.Close()
	buf := new(bytes.Buffer)

	_, err := buf.ReadFrom(obj)
	exception.PanicLogging(err, "Unable to create buffer")

	return buf
}
