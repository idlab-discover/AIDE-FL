package impl

import (
	"context"
	"io"
	"log/slog"

	"github.com/idlab-discover/AIDE-FL/server/server/server-logic/configuration"
	"github.com/idlab-discover/AIDE-FL/server/server/server-logic/exception"
	"github.com/idlab-discover/AIDE-FL/server/server/server-logic/repository"
	"github.com/minio/minio-go/v7"
)

type minioRepositoryImpl struct {
	*minio.Client
	configuration.Config
}

func NewMinioRepositoryImpl(minioClient *minio.Client, config configuration.Config) repository.MinioRepository {
	return &minioRepositoryImpl{Client: minioClient, Config: config}
}

func (repo *minioRepositoryImpl) InsertModel(srcFilename string, dstFilename string) {
	slog.Info("Uploading model to Minio", "file", srcFilename)

	_, err := repo.Client.FPutObject(context.Background(), repo.Config.Get("MINIO_MODELS_BUCKET"), dstFilename, srcFilename, minio.PutObjectOptions{})
	exception.PanicLogging(err, "Unable to upload", "model", srcFilename)
}

func (repo *minioRepositoryImpl) InsertDataset(srcFilename string, dstFilename string) {
	slog.Info("Uploading dataset to Minio", "file", srcFilename)

	_, err := repo.Client.FPutObject(context.Background(), repo.Config.Get("MINIO_DATASETS_BUCKET"), dstFilename, srcFilename, minio.PutObjectOptions{})
	exception.PanicLogging(err, "Unable to upload", "dataset", srcFilename)
}

func (repo *minioRepositoryImpl) InsertCheckpoint(srcFilename string, dstFilename string) {
	slog.Info("Uploading checkpoint to Minio", "file", srcFilename)

	_, err := repo.Client.FPutObject(context.Background(), repo.Config.Get("MINIO_CHECKPOINTS_BUCKET"), dstFilename, srcFilename, minio.PutObjectOptions{})
	exception.PanicLogging(err, "Unable to upload", "checkpoint", srcFilename)
}

func (repo *minioRepositoryImpl) InsertMetric(srcFilename string, dstFilename string) {
	slog.Info("Uploading metric to Minio", "file", srcFilename)

	_, err := repo.Client.FPutObject(context.Background(), repo.Config.Get("MINIO_METRICS_BUCKET"), dstFilename, srcFilename, minio.PutObjectOptions{})
	exception.PanicLogging(err, "Unable to upload", "metric", srcFilename)
}

func (repo *minioRepositoryImpl) ModelExists(filename string) bool {
	slog.Info("Checking Minio", "model", filename)

	_, err := repo.Client.StatObject(context.Background(), repo.Config.Get("MINIO_MODELS_BUCKET"), filename, minio.StatObjectOptions{})

	return err == nil
}

func (repo *minioRepositoryImpl) DatasetExists(filename string) bool {
	slog.Info("Checking Minio", "dataset", filename)

	_, err := repo.Client.StatObject(context.Background(), repo.Config.Get("MINIO_DATASETS_BUCKET"), filename, minio.StatObjectOptions{})

	return err == nil
}

func (repo *minioRepositoryImpl) BucketExists(bucketName string) bool {
	slog.Info("Checking Minio", "bucket", bucketName)

	exists, err := repo.Client.BucketExists(context.Background(), bucketName)
	exception.PanicLogging(err, "Unable to check for", "bucket", bucketName)

	return exists
}

func (repo *minioRepositoryImpl) GetModel(filename string) io.Reader {
	slog.Info("Getting Minio", "model", filename)

	obj, err := repo.Client.GetObject(context.Background(), repo.Config.Get("MINIO_MODELS_BUCKET"), filename, minio.GetObjectOptions{})
	exception.PanicLogging(err, "Unable to get", "model", filename)

	return obj
}

func (repo *minioRepositoryImpl) GetDataset(filename string) io.Reader {
	slog.Info("Getting Minio", "dataset", filename)

	obj, err := repo.Client.GetObject(context.Background(), repo.Config.Get("MINIO_DATASETS_BUCKET"), filename, minio.GetObjectOptions{})
	exception.PanicLogging(err, "Unable to get", "dataset", filename)

	return obj
}

func (repo *minioRepositoryImpl) CreateBucket(bucketName string) {
	slog.Info("Creating Minio", "bucket", bucketName)

	err := repo.Client.MakeBucket(context.Background(), bucketName, minio.MakeBucketOptions{})
	exception.PanicLogging(err, "Unable to create", "bucket", bucketName)
}
