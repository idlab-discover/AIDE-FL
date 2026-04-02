package controller

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/model"
	"github.com/MartienskieNum1/AIDE-FL/server/server/server-logic/service"
	"github.com/google/uuid"
)

type fakeServerLogicService struct {
	decodeScenarioFn    func(io.ReadCloser) (*model.Scenario, error)
	validateScenarioFn  func(*model.Scenario) error
	startClientsFn      func(*model.Scenario, uuid.UUID, map[string]*model.Client)
	startTrainingFn     func() (string, error)
	getTrainingStatusFn func(string) (string, error)
}

func (f *fakeServerLogicService) DecodeScenario(r io.ReadCloser) (*model.Scenario, error) {
	return f.decodeScenarioFn(r)
}

func (f *fakeServerLogicService) ValidateScenario(scenario *model.Scenario) error {
	return f.validateScenarioFn(scenario)
}

func (f *fakeServerLogicService) StartClients(scenario *model.Scenario, prepID uuid.UUID, clients map[string]*model.Client) {
	if f.startClientsFn != nil {
		f.startClientsFn(scenario, prepID, clients)
	}
}

func (f *fakeServerLogicService) SetFlowerConfig(_ *model.Scenario) {}

func (f *fakeServerLogicService) StartTraining() (string, error) {
	return f.startTrainingFn()
}

func (f *fakeServerLogicService) GetTrainingStatus(runID string) (string, error) {
	return f.getTrainingStatusFn(runID)
}

func (f *fakeServerLogicService) WatchForEndOfTraining(_ func(string), _ string) {}

func (f *fakeServerLogicService) RemoveArtifacts() {}

type fakeMinioService struct{}

func (f *fakeMinioService) EnsureBucketExists(string)                 {}
func (f *fakeMinioService) CreateDataset(string) error                { return nil }
func (f *fakeMinioService) CreateAndShareHoldOutDataset(string) error { return nil }
func (f *fakeMinioService) UploadArtifacts(string)                    {}

type fakeConfig struct{}

func (fakeConfig) Get(string) string { return "" }

type fakeGeneralMessageReader struct {
	readMessageFn func() (int, []byte, error)
}

func (f *fakeGeneralMessageReader) ReadMessage() (int, []byte, error) {
	return f.readMessageFn()
}

func newTestController(t *testing.T, server service.ServerLogicService, minio service.MinioService) *ServerLogicController {
	t.Helper()

	return &ServerLogicController{
		ServerLogicService:       server,
		MinioService:             minio,
		Config:                   fakeConfig{},
		clients:                  make(map[string]*model.Client),
		clientsMutex:             &sync.Mutex{},
		activeSessionMutex:       &sync.Mutex{},
		latestTrainingRunIdMutex: &sync.Mutex{},
	}
}

func TestStartRejectsConcurrentTraining(t *testing.T) {
	server := &fakeServerLogicService{
		decodeScenarioFn: func(r io.ReadCloser) (*model.Scenario, error) {
			t.Fatal("DecodeScenario should not be called when training is already active")
			return nil, nil
		},
		validateScenarioFn: func(*model.Scenario) error { return nil },
		startTrainingFn:    func() (string, error) { return "", nil },
		getTrainingStatusFn: func(string) (string, error) {
			return `{"success":true,"runs":[{"run-id":"run-123","status":"running"}]}`, nil
		},
	}
	minio := &fakeMinioService{}
	controller := newTestController(t, server, minio)
	controller.activeSession = newTrainingSession()
	controller.setLatestTrainingRunId("run-123")

	req := httptest.NewRequest(http.MethodPost, "/start", bytes.NewBufferString(`{}`))
	rec := httptest.NewRecorder()

	controller.start(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected status %d, got %d", http.StatusTooManyRequests, rec.Code)
	}

	var response model.GeneralResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("unable to decode response: %v", err)
	}
	if response.Message != "Aborting training due to already ongoing training run" {
		t.Fatalf("unexpected message: %q", response.Message)
	}
}

func TestStartReleasesTrainingSessionOnInvalidRequest(t *testing.T) {
	server := &fakeServerLogicService{
		decodeScenarioFn: func(io.ReadCloser) (*model.Scenario, error) {
			return nil, io.ErrUnexpectedEOF
		},
		validateScenarioFn: func(*model.Scenario) error {
			t.Fatal("ValidateScenario should not be called when decoding fails")
			return nil
		},
		startTrainingFn:     func() (string, error) { return "", nil },
		getTrainingStatusFn: func(string) (string, error) { return "", nil },
	}
	minio := &fakeMinioService{}
	controller := newTestController(t, server, minio)

	req := httptest.NewRequest(http.MethodPost, "/start", bytes.NewBufferString(`{`))
	rec := httptest.NewRecorder()

	controller.start(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
	if controller.currentTrainingSession() != nil {
		t.Fatal("expected training session to be released after invalid request")
	}
}

func TestTrainingSessionHandlesDisconnectAfterReconnect(t *testing.T) {
	session := newTrainingSession()
	oldClient := &model.Client{Id: "client-a", Address: "old"}
	newClient := &model.Client{Id: "client-a", Address: "new"}

	session.reset(map[string]*model.Client{
		oldClient.Id: oldClient,
	})
	session.handleClientDisconnect(newClient)

	if session.errorMessage == nil {
		t.Fatal("expected active client disconnect to abort preparation after reconnect")
	}
	if session.errorMessage.Origin != "server" {
		t.Fatalf("unexpected error origin: %q", session.errorMessage.Origin)
	}
}

func TestListenForGeneralMessagesReturnsOnReadError(t *testing.T) {
	controller := newTestController(t, &fakeServerLogicService{}, &fakeMinioService{})
	reader := &fakeGeneralMessageReader{
		readMessageFn: func() (int, []byte, error) {
			return 0, nil, errors.New("read failed")
		},
	}

	controller.listenForGeneralMessages(reader, "client-a")
}

func TestStartReturnsRetryableErrorWhenPreparationAborts(t *testing.T) {
	var controller *ServerLogicController

	server := &fakeServerLogicService{
		decodeScenarioFn: func(io.ReadCloser) (*model.Scenario, error) {
			return &model.Scenario{
				Name:      "scenario-a",
				Framework: "tensorflow",
				Aggregation: model.Aggregation{
					Strategy:            "fedavg",
					MinAvailableClients: 1,
				},
				TensorFlow: &model.TensorFlowFramework{Model: model.TensorFlowModel{Config: map[string]interface{}{"config": map[string]interface{}{"layers": []interface{}{}}}}},
			}, nil
		},
		validateScenarioFn: func(*model.Scenario) error { return nil },
		startClientsFn: func(_ *model.Scenario, _ uuid.UUID, _ map[string]*model.Client) {
			controller.currentTrainingSession().reportError(&model.GeneralMessage{
				Type:    "ERROR",
				Origin:  "server",
				Message: "client disconnected while preparing",
			})
		},
		startTrainingFn:     func() (string, error) { return "", nil },
		getTrainingStatusFn: func(string) (string, error) { return "", nil },
	}

	controller = newTestController(t, server, &fakeMinioService{})
	controller.clients["client-a"] = &model.Client{Id: "client-a"}

	req := httptest.NewRequest(http.MethodPost, "/start", bytes.NewBufferString(`{}`))
	rec := httptest.NewRecorder()

	controller.start(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, rec.Code)
	}
	if rec.Header().Get("Retry-After") != "5" {
		t.Fatalf("expected Retry-After header 5, got %q", rec.Header().Get("Retry-After"))
	}

	var response model.GeneralResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("unable to decode response: %v", err)
	}
	if response.Message != "Training run aborted due to client error" {
		t.Fatalf("unexpected message: %q", response.Message)
	}
}
