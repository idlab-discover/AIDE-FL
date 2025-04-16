package model

type Layer struct {
	ClassName string                 `json:"class_name"`
	Config    map[string]interface{} `json:"config"`
}
