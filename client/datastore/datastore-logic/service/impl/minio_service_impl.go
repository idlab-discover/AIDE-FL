package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/repository"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
	repository.LocalRepository
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository, localRepository *repository.LocalRepository) service.MinioService {
	return &minioServiceImpl{MinioRepository: *minioRepository, LocalRepository: *localRepository}
}

func (service *minioServiceImpl) Create(scenario model.Scenario) model.Scenario {

	datasetFilename := scenario.Dataset + ".parquet"

	if !service.MinioRepository.ObjectExists(datasetFilename) {
		zipName := service.LocalRepository.DownloadKaggleDataset(scenario.Dataset)
		service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename)
		service.MinioRepository.Insert(datasetFilename, datasetFilename)
		service.LocalRepository.RemoveFile(datasetFilename)
	}

	return scenario
}

func (service *minioServiceImpl) EnsureBucketExists(bucketName string) {
	if !service.MinioRepository.BucketExists(bucketName) {
		service.MinioRepository.CreateBucket(bucketName)
	}
}
