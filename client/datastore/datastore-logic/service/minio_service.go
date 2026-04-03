package service

import (
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/model"
)

type MinioService interface {
	Create(scenario model.Scenario) (model.Scenario, error)
	EnsureBucketExists(bucketName string)
}
