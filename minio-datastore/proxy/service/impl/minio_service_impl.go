package impl

import (
	"context"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
	repository.LocalRepository
	configuration.Config
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository, localRepository *repository.LocalRepository, config configuration.Config) service.MinioService {
	return &minioServiceImpl{MinioRepository: *minioRepository, LocalRepository: *localRepository, Config: config}
}

func (service *minioServiceImpl) Create(ctx context.Context, scenario model.Scenario) model.Status {

	datasetFilename := scenario.Dataset + ".parquet"
	scenarioFilename := scenario.Name + ".json"
	indexFilename := scenario.Name + ".INDEX"
	minioDatasetFilename := scenario.Dataset + "/" + datasetFilename
	minioScenarioFilename := scenario.Dataset + "/" + scenarioFilename
	minioIndexFilename := scenario.Dataset + "/" + indexFilename

	var status model.Status

	if service.MinioRepository.ObjectExists(minioIndexFilename) {
		status.Status = "preprocessed"
	} else {
		if !service.MinioRepository.ObjectExists(minioScenarioFilename) {

			service.LocalRepository.ScenarioToFile(scenarioFilename, scenario)
			service.MinioRepository.Insert(scenarioFilename, minioScenarioFilename)
			service.LocalRepository.RemoveFile(scenarioFilename)

			status.Status = "new scenario"
		} else {
			status.Status = "not preprocessed"
		}

		if !service.MinioRepository.ObjectExists(minioDatasetFilename) {
			zipName := service.LocalRepository.DownloadKaggleDataset(scenario.Dataset)
			service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename)
			service.MinioRepository.Insert(datasetFilename, minioDatasetFilename)
			service.LocalRepository.RemoveFile(datasetFilename)

			status.Status = "new dataset"
		}
	}

	return status
}

func (service *minioServiceImpl) EnsureBucketExists(bucketName string) {
	if !service.MinioRepository.BucketExists(bucketName) {
		service.MinioRepository.CreateBucket(bucketName)
	}
}
