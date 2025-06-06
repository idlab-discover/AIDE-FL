package service

import (
	"github.com/MartienskieNum1/AIDE-FL/client/datastore/proxy/model"
)

type MinioService interface {
	Create(scenario model.Scenario) model.Scenario
	EnsureBucketExists(bucketName string)
}
