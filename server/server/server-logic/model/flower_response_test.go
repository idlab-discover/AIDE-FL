package model

import (
	"encoding/json"
	"testing"
)

func TestFlowerResponseAllowsNumericElapsed(t *testing.T) {
	t.Parallel()

	payload := []byte(`{
		"success": true,
		"runs": [{
			"run-id": "123",
			"fab-id": "flower/flower",
			"fab-name": "flower",
			"fab-version": "1.0.0",
			"fab-hash": "abc123",
			"status": "running",
			"elapsed": 2.345,
			"created-at": "2026-03-31T15:12:39Z",
			"running-at": "2026-03-31T15:12:40Z",
			"finished-at": ""
		}]
	}`)

	var response FlowerResponse
	if err := json.Unmarshal(payload, &response); err != nil {
		t.Fatalf("unmarshal numeric elapsed: %v", err)
	}

	if got := string(response.Runs[0].Elapsed); got != "2.345" {
		t.Fatalf("elapsed = %q, want %q", got, "2.345")
	}
}

func TestFlowerResponseAllowsStringElapsed(t *testing.T) {
	t.Parallel()

	payload := []byte(`{
		"success": true,
		"runs": [{
			"run-id": "123",
			"fab-id": "flower/flower",
			"fab-name": "flower",
			"fab-version": "1.0.0",
			"fab-hash": "abc123",
			"status": "finished:completed",
			"elapsed": "00:01:34",
			"created-at": "2026-03-31T15:12:39Z",
			"running-at": "2026-03-31T15:12:40Z",
			"finished-at": "2026-03-31T15:14:13Z"
		}]
	}`)

	var response FlowerResponse
	if err := json.Unmarshal(payload, &response); err != nil {
		t.Fatalf("unmarshal string elapsed: %v", err)
	}

	if got := string(response.Runs[0].Elapsed); got != `"00:01:34"` {
		t.Fatalf("elapsed = %q, want %q", got, `"00:01:34"`)
	}
}
