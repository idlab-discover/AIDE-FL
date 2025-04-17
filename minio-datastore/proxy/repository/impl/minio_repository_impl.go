package impl

import (
	"context"
	"io"

	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/configuration"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/exception"
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/repository"
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
	_, err := repo.Client.FPutObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), dstFilename, srcFilename, minio.PutObjectOptions{})
	exception.PanicLogging(err, "Unable to upload", "object", srcFilename)
}

func (repo *minioRepositoryImpl) ObjectExists(filename string) bool {
	_, err := repo.Client.StatObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), filename, minio.StatObjectOptions{})

	return err == nil
}

func (repo *minioRepositoryImpl) BucketExists(bucketName string) bool {
	exists, err := repo.Client.BucketExists(context.Background(), repo.Config.Get("MINIO_BUCKET"))
	exception.PanicLogging(err, "Unable to check for", "bucket", bucketName)

	return exists
}

func (repo *minioRepositoryImpl) GetObject(filename string) io.Reader {
	obj, err := repo.Client.GetObject(context.Background(), repo.Config.Get("MINIO_BUCKET"), filename, minio.GetObjectOptions{})
	exception.PanicLogging(err, "Unable to get", "object", filename)
	defer obj.Close()

	return obj
}

func (repo *minioRepositoryImpl) CreateBucket(bucketName string) {
	err := repo.Client.MakeBucket(context.Background(), repo.Config.Get("MINIO_BUCKET"), minio.MakeBucketOptions{})
	exception.PanicLogging(err, "Unable to create", "bucket", bucketName)
}
