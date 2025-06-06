package impl

import (
	"path/filepath"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/repository"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository) service.MinioService {
	return &minioServiceImpl{MinioRepository: *minioRepository}
}

func (service *minioServiceImpl) EnsureBucketExists(bucketName string) {
	if !service.MinioRepository.BucketExists(bucketName) {
		service.MinioRepository.CreateBucket(bucketName)
	}
}

func (service *minioServiceImpl) UploadSafetensors(path string) {
	service.MinioRepository.Insert(path, filepath.Base(path))
}
