package impl

import (
	"archive/zip"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/repository"
)

type localRepositoryImpl struct {
	configuration.Config
}

func NewLocalRepositoryImpl(config configuration.Config) repository.LocalRepository {
	return &localRepositoryImpl{Config: config}
}

func (repo *localRepositoryImpl) RemoveFile(filename string) {
	slog.Info("Removing", "file", filename)

	err := os.Remove(filename)
	exception.PanicLogging(err, "Unable to remove", "file", filename)
}

func (repo *localRepositoryImpl) ExtractParquetFromZip(zipFilename string, outFilename string) {
	slog.Info("Extracting first parquet file from", "zip", zipFilename)

	archive, err := zip.OpenReader(zipFilename)
	exception.PanicLogging(err, "Unable to extract from", "file", zipFilename)
	defer archive.Close()

	// Extract first parquet file
	for _, f := range archive.File {
		if strings.HasSuffix(f.Name, ".parquet") {
			fileInArchive, err := f.Open()
			exception.PanicLogging(err, "Unable to open", "file", f.Name)

			dstFile, err := os.Create(outFilename)
			exception.PanicLogging(err, "Unable to create", "file", outFilename)

			_, err = io.Copy(dstFile, fileInArchive)
			exception.PanicLogging(err, "Unable to write", "file", dstFile)

			dstFile.Close()
			fileInArchive.Close()

			break
		}
	}

	// Remove zip after extraction
	repo.RemoveFile(zipFilename)
}

func (repo *localRepositoryImpl) DownloadKaggleDataset(dataset string) (string, error) {
	slog.Info("Downloading Kaggle", "dataset", dataset)

	if !strings.Contains(dataset, "-") {
		return "", errors.New("dataset name must contain a '-' to split kaggle username from kaggle dataset")
	}

	fileKeyURL := repo.Config.Get("KAGGLE_URL") + strings.Replace(dataset, "-", "/", -1)

	// Download from Kaggle
	resp, err := http.Get(fileKeyURL)
	exception.PanicLogging(err, "Unable to download from Kaggle", "dataset", fileKeyURL)
	defer resp.Body.Close()

	// Dataset not reachable on Kaggle
	if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusNotFound {
		return "", errors.New("Kaggle dataset not found with URL " + fileKeyURL)
	}

	//TODO: use WriteFile
	outFilename := dataset + ".zip"
	file, err := os.Create(outFilename)
	exception.PanicLogging(err, "Unable to create", "file", outFilename)
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	exception.PanicLogging(err, "Unable to write", "file", file)

	return outFilename, nil
}
