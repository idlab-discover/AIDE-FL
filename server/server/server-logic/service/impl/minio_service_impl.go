package impl

import (
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/exception"
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
	holdOutPath := path.Join(shareDir, "hold-out.parquet")

	if !service.MinioRepository.DatasetExists(datasetFilename) {
		zipName, err := service.LocalRepository.DownloadKaggleDataset(dataset)
		if err != nil {
			return err
		}

		service.LocalRepository.ExtractParquetFromZip(zipName, datasetFilename)
		service.MinioRepository.InsertDataset(datasetFilename, datasetFilename)
		service.LocalRepository.MoveFile(datasetFilename, holdOutPath)
	} else {
		reader := service.MinioRepository.GetDataset(datasetFilename)
		service.LocalRepository.SaveToFile(reader, holdOutPath)
	}

	return nil
}

func (service *minioServiceImpl) UploadArtifacts(scenario_name string) {
	dir := shareDir

	files, err := os.ReadDir(dir)
	exception.PanicLogging(err, "Unable to get artifacts")

	// Get metrics
	var matched_metrics []string
	for _, entry := range files {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") {
			matched_metrics = append(matched_metrics, path.Join(dir, entry.Name()))
		}
	}

	for _, f := range matched_metrics {
		service.MinioRepository.InsertMetric(f, filepath.Base(f))
	}

	// Get checkpoints
	var matched_checkp []string
	for _, entry := range files {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".weights.h5") {
			matched_checkp = append(matched_checkp, path.Join(dir, entry.Name()))
		}
	}

	for _, f := range matched_checkp {
		service.MinioRepository.InsertCheckpoint(f, filepath.Base(f))
	}

	// Get final model
	service.MinioRepository.InsertModel(path.Join(shareDir, scenario_name+".safetensors"), scenario_name+".safetensors")
}
