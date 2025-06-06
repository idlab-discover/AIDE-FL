package model

type GeneralResponse struct {
	Message string      `json:"message"`
	Data    interface{} `json:"data"`
}
