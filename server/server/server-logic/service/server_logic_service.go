package service

import "github.com/MartienskieNum1/AIDE-FL/server/server-logic/model"

type ServerLogicService interface {
	StartClients(scenario model.Scenario, clients map[string]*model.Client)
	SetFlowerConfig(scenario model.Scenario)
	StartTraining() string
	GetTrainingStatus(runId string) string
}
