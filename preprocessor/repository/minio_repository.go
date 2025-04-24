package repository

import "io"

type MinioRepository interface {
	Insert(srcFilename string, dstFilename string)
	GetObject(filename string) io.ReadCloser
}
