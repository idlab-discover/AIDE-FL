package controller

import (
	"context"
	"errors"
	"testing"

	"github.com/idlab-discover/AIDE-FL/client/orchestrator/model"
)

type fakeGeneralMessageWriter struct {
	writeMessageFn func(messageType int, data []byte) error
}

func (f *fakeGeneralMessageWriter) WriteMessage(messageType int, data []byte) error {
	return f.writeMessageFn(messageType, data)
}

type fakeKafkaService struct {
	consumeGeneralMessageFn func(ctx context.Context) (model.GeneralMessage, error)
}

func (f *fakeKafkaService) PublishPrepareScenario(model.Scenario) {}

func (f *fakeKafkaService) PublishPreprocessScenario(model.Scenario) {}

func (f *fakeKafkaService) PublishTrainScenario(model.Scenario) {}

func (f *fakeKafkaService) ConsumePreparedScenario(context.Context, string) (model.Scenario, error) {
	return model.Scenario{}, nil
}

func (f *fakeKafkaService) ConsumePreprocessedScenario(context.Context, string) (model.Scenario, error) {
	return model.Scenario{}, nil
}

func (f *fakeKafkaService) ConsumeTrainedScenario(context.Context, string) (model.Scenario, error) {
	return model.Scenario{}, nil
}

func (f *fakeKafkaService) ConsumeGeneralMessage(ctx context.Context) (model.GeneralMessage, error) {
	return f.consumeGeneralMessageFn(ctx)
}

func TestHandleErrorMessageCancelsMatchingPreparation(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{}
	cancelled := false

	controller.setActiveScenario("active-prep", func() {
		cancelled = true
	})

	controller.handleErrorMessage(model.GeneralMessage{
		PrepID: "active-prep",
		Type:   "ERROR",
		Origin: "trainer",
	})

	if !cancelled {
		t.Fatal("handleErrorMessage did not cancel the active scenario")
	}

	if controller.activePrepID != "" {
		t.Fatalf("active prep id = %q, want empty", controller.activePrepID)
	}

	if controller.cancelScenarioFunc != nil {
		t.Fatal("cancelScenarioFunc was not cleared")
	}
}

func TestHandleErrorMessageIgnoresStalePreparation(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{}
	cancelled := false

	controller.setActiveScenario("active-prep", func() {
		cancelled = true
	})

	controller.handleErrorMessage(model.GeneralMessage{
		PrepID: "stale-prep",
		Type:   "ERROR",
		Origin: "trainer",
	})

	if cancelled {
		t.Fatal("handleErrorMessage cancelled the active scenario for a stale prep id")
	}

	if controller.activePrepID != "active-prep" {
		t.Fatalf("active prep id = %q, want %q", controller.activePrepID, "active-prep")
	}

	if controller.cancelScenarioFunc == nil {
		t.Fatal("cancelScenarioFunc was cleared unexpectedly")
	}
}

func TestHandleErrorMessageIgnoresMissingPrepID(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{}
	cancelled := false

	controller.setActiveScenario("active-prep", func() {
		cancelled = true
	})

	controller.handleErrorMessage(model.GeneralMessage{
		Type:   "ERROR",
		Origin: "trainer",
	})

	if cancelled {
		t.Fatal("handleErrorMessage cancelled the active scenario without a prep id")
	}

	if controller.activePrepID != "active-prep" {
		t.Fatalf("active prep id = %q, want %q", controller.activePrepID, "active-prep")
	}
}

func TestClearActiveScenarioDoesNotClearNewerPreparation(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{}
	controller.setActiveScenario("new-prep", func() {})

	controller.clearActiveScenario("old-prep")

	if controller.activePrepID != "new-prep" {
		t.Fatalf("active prep id = %q, want %q", controller.activePrepID, "new-prep")
	}

	if controller.cancelScenarioFunc == nil {
		t.Fatal("cancelScenarioFunc was cleared unexpectedly")
	}
}

func TestSetActiveScenarioStoresCancelFunc(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{}
	cancelled := false

	controller.setActiveScenario("prep-id", func() {
		cancelled = true
	})

	if controller.activePrepID != "prep-id" {
		t.Fatalf("active prep id = %q, want %q", controller.activePrepID, "prep-id")
	}

	if controller.cancelScenarioFunc == nil {
		t.Fatal("cancelScenarioFunc is nil")
	}

	controller.cancelScenarioFunc()

	if !cancelled {
		t.Fatal("cancelScenarioFunc did not call the provided cancel function")
	}
}

func TestConsumeGeneralMessagesReturnsWriteFailure(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{
		KafkaService: &fakeKafkaService{
			consumeGeneralMessageFn: func(context.Context) (model.GeneralMessage, error) {
				return model.GeneralMessage{
					Type:   "INFO",
					Origin: "trainer",
				}, nil
			},
		},
	}

	writer := &fakeGeneralMessageWriter{
		writeMessageFn: func(int, []byte) error {
			return errors.New("socket closed")
		},
	}

	err := controller.consumeGeneralMessages(context.Background(), writer)
	if err == nil {
		t.Fatal("consumeGeneralMessages returned nil error")
	}
	if err.Error() != "unable to send general message to server: socket closed" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestConsumeGeneralMessagesReturnsNilWhenCanceled(t *testing.T) {
	t.Helper()

	controller := &OrchestratorController{
		KafkaService: &fakeKafkaService{
			consumeGeneralMessageFn: func(context.Context) (model.GeneralMessage, error) {
				return model.GeneralMessage{}, context.Canceled
			},
		},
	}

	err := controller.consumeGeneralMessages(context.Background(), &fakeGeneralMessageWriter{})
	if err != nil {
		t.Fatalf("consumeGeneralMessages returned unexpected error: %v", err)
	}
}
