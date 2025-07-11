package impl

import (
	"context"
	"io"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/configuration"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/datastore-logic/repository"
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

func (repo *minioRepositoryImpl) ObjectExists(filename string) bool {
	slog.Info("Checking Minio", "object", filename)

	_, err := repo.Client.StatObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), filename, minio.StatObjectOptions{})

	return err == nil
}

func (repo *minioRepositoryImpl) BucketExists(bucketName string) bool {
	slog.Info("Checking Minio", "bucket", bucketName)

	exists, err := repo.Client.BucketExists(context.Background(), bucketName)
	exception.PanicLogging(err, "Unable to check for", "bucket", bucketName)

	return exists
}

func (repo *minioRepositoryImpl) GetObject(filename string) io.Reader {
	slog.Info("Getting Minio", "object", filename)

	obj, err := repo.Client.GetObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), filename, minio.GetObjectOptions{})
	exception.PanicLogging(err, "Unable to get", "object", filename)
	defer obj.Close()

	return obj
}

func (repo *minioRepositoryImpl) CreateBucket(bucketName string) {
	slog.Info("Creating Minio", "bucket", bucketName)

	err := repo.Client.MakeBucket(context.Background(), bucketName, minio.MakeBucketOptions{})
	exception.PanicLogging(err, "Unable to create", "bucket", bucketName)
}
