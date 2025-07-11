package repository

type LocalRepository interface {
	RemoveFile(filename string)
	ExtractParquetFromZip(zipFilename string, outFilename string)
	DownloadKaggleDataset(dataset string) (string, error)
}
