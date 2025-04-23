package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/repository"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository) service.MinioService {
	return &minioServiceImpl{MinioRepository: *minioRepository}
}
