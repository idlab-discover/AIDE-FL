package repository

import "github.com/MartienskieNum1/AIDE-FL/minio-datastore/proxy/model"

type LocalRepository interface {
	ScenarioToFile(outFilename string, scenario model.Scenario)
	RemoveFile(filename string)
	ExtractParquetFromZip(zipFilename string, outFilename string)
	DownloadKaggleDataset(dataset string) string
}
