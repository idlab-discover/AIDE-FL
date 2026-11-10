package service

import (
	"io"

	"github.com/google/uuid"
	"github.com/idlab-discover/AIDE-FL/server/server/server-logic/model"
)

type ServerLogicService interface {
	DecodeScenario(r io.ReadCloser) (*model.Scenario, error)
	ValidateScenario(scenario *model.Scenario) error
	StartClients(scenario *model.Scenario, prepId uuid.UUID, clients map[string]*model.Client)
	SetFlowerConfig(scenario *model.Scenario)
	StartTraining() (string, error)
	GetTrainingStatus(runId string) (string, error)
	WatchForEndOfTraining(callback func(string), scenario_name string)
	RemoveArtifacts()
}
