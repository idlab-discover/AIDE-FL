package configuration

import (
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

func NewMinioClient(config Config) *minio.Client {
	username := config.Get("MINIO_ACCESS_KEY")
	password := config.Get("MINIO_SECRET_KEY")
	endpoint := config.Get("MINIO_ENDPOINT")

	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(username, password, ""),
		Secure: false,
	})
	exception.PanicLogging(err, "Unable to create Minio client")

	return client
}
