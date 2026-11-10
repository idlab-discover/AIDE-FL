package service

import (
	"github.com/idlab-discover/AIDE-FL/client/datastore/datastore-logic/model"
)

type MinioService interface {
	Create(scenario model.Scenario) (model.Scenario, error)
	EnsureBucketExists(bucketName string)
}
