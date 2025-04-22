package service

import (
	"github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"
)

type MinioService interface {
	Create(scenario model.Scenario) model.Status
	EnsureBucketExists(bucketName string)
}
