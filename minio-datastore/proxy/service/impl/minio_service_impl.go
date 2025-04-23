package impl

import (
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/service"
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
	scenarioFilename := scenario.Name + ".json"
	indexFilename := scenario.Name + ".INDEX"
	minioDatasetFilename := scenario.Dataset + "/" + datasetFilename
	minioScenarioFilename := scenario.Dataset + "/" + scenarioFilename
	minioIndexFilename := scenario.Dataset + "/" + indexFilename

	if service.MinioRepository.ObjectExists(minioIndexFilename) {
		scenario.Status = "preprocessed"
	} else {
		if !service.MinioRepository.ObjectExists(minioScenarioFilename) {

			service.LocalRepository.ScenarioToFile(scenarioFilename, scenario)
			service.MinioRepository.Insert(scenarioFilename, minioScenarioFilename)
			service.LocalRepository.RemoveFile(scenarioFilename)

			scenario.Status = "new scenario"
		} else {
			scenario.Status = "not preprocessed"
		}

		if !service.MinioRepository.ObjectExists(minioDatasetFilename) {
			zipName := service.LocalRepository.DownloadKaggleDataset(scenario.Dataset)
			service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename)
			service.MinioRepository.Insert(datasetFilename, minioDatasetFilename)
			service.LocalRepository.RemoveFile(datasetFilename)

			scenario.Status = "new dataset"
		}
	}

	return scenario
}

func (service *minioServiceImpl) EnsureBucketExists(bucketName string) {
	if !service.MinioRepository.BucketExists(bucketName) {
		service.MinioRepository.CreateBucket(bucketName)
	}
}
