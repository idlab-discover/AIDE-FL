package service

import "github.com/MartienskieNum1/AIDE-FL/server/server-logic/model"

type ServerLogicService interface {
	StartClients(scenario model.Scenario)
	SetFlowerConfig(scenario model.Scenario)
}
