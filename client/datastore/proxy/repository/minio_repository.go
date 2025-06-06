package repository

import "io"

type MinioRepository interface {
	Insert(srcFilename string, dstFilename string)
	ObjectExists(filename string) bool
	BucketExists(bucketName string) bool
	GetObject(filename string) io.Reader
	CreateBucket(bucketName string)
}
