package repository

import "io"

type LocalRepository interface {
	RemoveFile(filename string)
	ExtractParquetFromZip(zipFilename string, outFilename string)
	DownloadKaggleDataset(dataset string) (string, error)
	MoveFile(srcFilename string, dstFilename string)
	SaveToFile(reader io.Reader, dstPath string)
}
