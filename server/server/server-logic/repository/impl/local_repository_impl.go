package impl

import (
	"archive/zip"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/configuration"
	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/exception"
	"github.com/Str-Gen/AIDE-FL/server/server/server-logic/repository"
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

func (repo *localRepositoryImpl) MoveFile(srcFilename string, dstFilename string) {
	slog.Info("Moving", "srcFilename", srcFilename, "dstFilename", dstFilename)

	err := os.Rename(srcFilename, dstFilename)
	if err == nil {
		return
	}

	var linkErr *os.LinkError
	if errors.As(err, &linkErr) && errors.Is(linkErr.Err, syscall.EXDEV) {
		slog.Info("Rename crossed filesystems, falling back to copy", "srcFilename", srcFilename, "dstFilename", dstFilename)

		repo.copyFile(srcFilename, dstFilename)
		repo.RemoveFile(srcFilename)
		return
	}

	exception.PanicLogging(err, "Unable to move a file", "srcFilename", srcFilename, "dstFilename", dstFilename)
}

func (repo *localRepositoryImpl) copyFile(srcFilename string, dstFilename string) {
	srcFile, err := os.Open(srcFilename)
	exception.PanicLogging(err, "Unable to open source file", "srcFilename", srcFilename)
	defer srcFile.Close()

	err = os.MkdirAll(filepath.Dir(dstFilename), 0o755)
	exception.PanicLogging(err, "Unable to create destination directory", "dstFilename", dstFilename)

	dstFile, err := os.Create(dstFilename)
	exception.PanicLogging(err, "Unable to create destination file", "dstFilename", dstFilename)
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	exception.PanicLogging(err, "Unable to copy moved file", "srcFilename", srcFilename, "dstFilename", dstFilename)
}

func (repo *localRepositoryImpl) SaveToFile(reader io.Reader, dstPath string) {
	slog.Info("Saving new file", "dstPath", dstPath)

	err := os.MkdirAll(filepath.Dir(dstPath), 0o755)
	exception.PanicLogging(err, "Unable to create file directory", "dstPath", dstPath)

	// Create new file
	outFile, err := os.Create(dstPath)
	exception.PanicLogging(err, "Unable to create new file", "dstPath", dstPath)
	defer outFile.Close()

	// Copy data from reader to new file
	_, err = io.Copy(outFile, reader)
	exception.PanicLogging(err, "Unable to copy data to new file", "dstPath", dstPath)
}
