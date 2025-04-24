package service

import "bytes"

type MinioService interface {
	LoadDataset(dataset string) *bytes.Buffer
}
