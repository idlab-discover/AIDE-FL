package service

import (
	"io"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/model"
)

type ServerLogicService interface {
	DecodeScenario(r io.ReadCloser) (*model.Scenario, error)
	ValidateScenario(scenario *model.Scenario) error
	StartClients(scenario *model.Scenario, clients map[string]*model.Client)
	SetFlowerConfig(scenario *model.Scenario)
	StartTraining() string
	GetTrainingStatus(runId string) string
	WatchForSafetensors(callback func(string))
}
