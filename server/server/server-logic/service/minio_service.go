package service

type MinioService interface {
	EnsureBucketExists(bucketName string)
	CreateDataset(dataset string) error
	UploadSafetensors(path string)
}
