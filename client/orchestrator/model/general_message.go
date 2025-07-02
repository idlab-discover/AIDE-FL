package model

type GeneralMessage struct {
	Type    string `json:"type"`
	Origin  string `json:"origin"`
	Message string `json:"message"`
}
