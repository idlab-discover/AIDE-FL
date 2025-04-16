package impl

import (
	"archive/zip"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
)

type localRepositoryImpl struct {
	configuration.Config
}

func NewLocalRepositoryImpl(config configuration.Config) repository.LocalRepository {
	return &localRepositoryImpl{Config: config}
}

func (repo *localRepositoryImpl) ScenarioToFile(outFilename string, scenario model.Scenario) {
	data, err := json.Marshal(scenario)
	exception.PanicLogging(err)

	err = os.WriteFile(outFilename, []byte(data), 0644)
	exception.PanicLogging(err)
}

func (repo *localRepositoryImpl) RemoveFile(filename string) {
	err := os.Remove(filename)
	exception.PanicLogging(err)
}

func (repo *localRepositoryImpl) ExtractParquetFromZip(zipFilename string, outFilename string) {
	archive, err := zip.OpenReader(zipFilename)
	exception.PanicLogging(err)
	defer archive.Close()

	// Extract first parquet file
	for _, f := range archive.File {
		if strings.HasSuffix(f.Name, ".parquet") {
			fileInArchive, err := f.Open()
			exception.PanicLogging(err)

			dstFile, err := os.Create(outFilename)
			exception.PanicLogging(err)

			_, err = io.Copy(dstFile, fileInArchive)
			exception.PanicLogging(err)

			dstFile.Close()
			fileInArchive.Close()

			break
		}
	}

	// Remove zip after extraction
	repo.RemoveFile(zipFilename)
}

func (repo *localRepositoryImpl) DownloadKaggleDataset(dataset string) string {
	fileKeyURL := repo.Config.Get("KAGGLE_URL") + strings.Replace(dataset, "-", "/", -1)

	// Downloaden van Kaggle
	resp, err := http.Get(fileKeyURL)
	exception.PanicLogging(err)
	defer resp.Body.Close()

	// TODO: error handling
	if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusNotFound {
		panic("Dataset not found")
	}

	//TODO: use WriteFile
	outName := dataset + ".zip"
	file, err := os.Create(outName)
	exception.PanicLogging(err)
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	exception.PanicLogging(err)

	return outName
}
