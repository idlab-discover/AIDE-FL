package impl

import (
	"github.com/idlab-discover/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/idlab-discover/AIDE-FL/client/datastore/datastore-logic/repository"
	"github.com/idlab-discover/AIDE-FL/client/datastore/datastore-logic/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
	repository.LocalRepository
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository, localRepository *repository.LocalRepository) service.MinioService {
	return &minioServiceImpl{MinioRepository: *minioRepository, LocalRepository: *localRepository}
}

func (service *minioServiceImpl) Create(scenario model.Scenario) (model.Scenario, error) {

	datasetFilename := scenario.Dataset + ".parquet"

	if !service.MinioRepository.ObjectExists(datasetFilename) {
		zipName, err := service.LocalRepository.DownloadKaggleDataset(scenario.Dataset)
		if err != nil {
			return model.Scenario{}, err
		}

		service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename, scenario.DatasetFile)
		service.MinioRepository.Insert(datasetFilename, datasetFilename)
		service.LocalRepository.RemoveFile(datasetFilename)
	}

	return scenario, nil
}

func (service *minioServiceImpl) EnsureBucketExists(bucketName string) {
	if !service.MinioRepository.BucketExists(bucketName) {
		service.MinioRepository.CreateBucket(bucketName)
	}
}
