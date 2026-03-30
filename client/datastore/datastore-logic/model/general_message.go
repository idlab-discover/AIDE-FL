package model

type GeneralMessage struct {
	PrepID  string `json:"prep_id,omitempty"`
	Type    string `json:"type"`
	Origin  string `json:"origin"`
	Message string `json:"message"`
}
