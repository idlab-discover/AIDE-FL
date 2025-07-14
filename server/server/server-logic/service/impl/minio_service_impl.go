package impl

import (
	"path"
	"path/filepath"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/repository"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service"
)

type minioServiceImpl struct {
	repository.MinioRepository
	repository.LocalRepository
}

func NewMinioServiceImpl(minioRepository *repository.MinioRepository, localRepository *repository.LocalRepository) service.MinioService {
	return &minioServiceImpl{
		MinioRepository: *minioRepository,
		LocalRepository: *localRepository,
	}
}

func (service *minioServiceImpl) EnsureBucketExists(bucketName string) {
	if !service.MinioRepository.BucketExists(bucketName) {
		service.MinioRepository.CreateBucket(bucketName)
	}
}

func (service *minioServiceImpl) CreateDataset(dataset string) error {

	datasetFilename := dataset + ".parquet"

	if !service.MinioRepository.DatasetExists(datasetFilename) {
		zipName, err := service.LocalRepository.DownloadKaggleDataset(dataset)
		if err != nil {
			return err
		}

		service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename)
		service.MinioRepository.InsertDataset(datasetFilename, datasetFilename)
		service.LocalRepository.RemoveFile(datasetFilename)
	}

	return nil
}

func (service *minioServiceImpl) CreateAndShareHoldOutDataset(dataset string) error {
	datasetFilename := dataset + ".parquet"

	if !service.MinioRepository.DatasetExists(datasetFilename) {
		zipName, err := service.LocalRepository.DownloadKaggleDataset(dataset)
		if err != nil {
			return err
		}

		service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename)
		service.MinioRepository.InsertDataset(datasetFilename, datasetFilename)
		service.LocalRepository.MoveFile(datasetFilename, path.Join("app", "share", "hold-out.parquet"))
	} else {
		reader := service.MinioRepository.GetDataset(datasetFilename)
		service.LocalRepository.SaveToFile(reader, path.Join("app", "share", "hold-out.parquet"))
	}

	return nil
}

func (service *minioServiceImpl) UploadSafetensors(path string) {
	service.MinioRepository.InsertModel(path, filepath.Base(path))
}
