package configuration

import (
	"os"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/joho/godotenv"
)

type Config interface {
	Get(key string) string
}

type configImpl struct {
}

func (config *configImpl) Get(key string) string {
	return os.Getenv(key)
}

func New(filenames ...string) Config {
	err := godotenv.Load(filenames...)
	if err != nil && !os.IsNotExist(err) {
		exception.PanicLogging(err, "Unable to load", "filenames", filenames)
	}
	return &configImpl{}
}
