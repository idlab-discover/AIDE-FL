package impl

import (
	"context"
	"io"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/preprocessor/configuration"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/exception"
	"github.com/MartienskieNum1/AIDE-FL/preprocessor/repository"
	"github.com/minio/minio-go/v7"
)

type minioRepositoryImpl struct {
	*minio.Client
	configuration.Config
}

func NewMinioRepositoryImpl(minioClient *minio.Client, config configuration.Config) repository.MinioRepository {
	return &minioRepositoryImpl{Client: minioClient, Config: config}
}

func (repo *minioRepositoryImpl) Insert(srcFilename string, dstFilename string) {
	slog.Info("Uploading to Minio", "file", srcFilename)

	_, err := repo.Client.FPutObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), dstFilename, srcFilename, minio.PutObjectOptions{})
	exception.PanicLogging(err, "Unable to upload", "object", srcFilename)
}

func (repo *minioRepositoryImpl) GetObject(filename string) io.ReadCloser {
	slog.Info("Getting Minio", "object", filename)

	obj, err := repo.Client.GetObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), filename, minio.GetObjectOptions{})
	exception.PanicLogging(err, "Unable to get", "object", filename)

	return obj
}
