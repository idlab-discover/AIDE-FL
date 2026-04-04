package service

type MinioService interface {
	EnsureBucketExists(bucketName string)
	CreateDataset(dataset string, datasetFile string) error
	CreateAndShareHoldOutDataset(dataset string, datasetFile string) error
	UploadArtifacts(path string)
}
