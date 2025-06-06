package service

type MinioService interface {
	EnsureBucketExists(bucketName string)
	UploadSafetensors(path string)
}
