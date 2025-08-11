package repository

import "io"

type MinioRepository interface {
	InsertModel(srcFilename string, dstFilename string)
	InsertDataset(srcFilename string, dstFilename string)
	InsertCheckpoint(srcFilename string, dstFilename string)
	InsertMetric(srcFilename string, dstFilename string)
	ModelExists(filename string) bool
	DatasetExists(filename string) bool
	BucketExists(bucketName string) bool
	GetModel(filename string) io.Reader
	GetDataset(filename string) io.Reader
	CreateBucket(bucketName string)
}
