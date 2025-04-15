package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
)

type Scenario struct {
	Name          string        `json:"name"`
	Dataset       string        `json:"dataset"`
	Preprocessing Preprocessing `json:"preprocessing"`
	Model         []Layer       `json:"model"`
}

type Preprocessing struct {
	ExcludeColumns []string `json:"exclude_columns"`
	TrainTestSplit float64  `json:"train_test_split"`
	Scaling        string   `json:"scaling"`
	Normalize      bool     `json:"normalize"`
}

type Layer struct {
	ClassName string                 `json:"class_name"`
	Config    map[string]interface{} `json:"config"`
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check Content-Type
	if r.Header.Get("Content-Type") != "application/json" {
		http.Error(w, "Expected Content-Type application/json", http.StatusUnsupportedMediaType)
		return
	}

	// Read raw JSON body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Could not read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parse JSON
	var scenario Scenario
	err = json.Unmarshal(body, &scenario)
	if err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	// Log parsed data
	log.Printf("Received Playbook:\nDataset: %s\nPreprocessing: %+v\nModel: %+v\n",
		scenario.Dataset, scenario.Preprocessing, scenario.Model)

	// Respond success
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Scenario parsed successfully"))
}

func main() {
	http.HandleFunc("/upload", uploadHandler)

	log.Println("Listening on http://localhost:8080/upload")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
