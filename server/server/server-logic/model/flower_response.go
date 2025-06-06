package model

type FlowerResponse struct {
	Success bool  `json:"success"`
	Runs    []Run `json:"runs"`
}

type Run struct {
	RunId      string `json:"run-id"`
	FabId      string `json:"fab-id"`
	FabName    string `json:"fab-name"`
	FabVersion string `json:"fab-version"`
	FabHash    string `json:"fab-hash"`
	Status     string `json:"status"`
	Elapsed    string `json:"elapsed"`
	CreatedAt  string `json:"created-at"`
	RunningAt  string `json:"running-at"`
	FinishedAt string `json:"finished-at"`
}
