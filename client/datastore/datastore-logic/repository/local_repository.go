package repository

import "github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/model"

type LocalRepository interface {
	ScenarioToFile(outFilename string, scenario model.Scenario)
	RemoveFile(filename string)
	ExtractParquetFromZip(zipFilename string, outFilename string)
	DownloadKaggleDataset(dataset string) (string, error)
}
