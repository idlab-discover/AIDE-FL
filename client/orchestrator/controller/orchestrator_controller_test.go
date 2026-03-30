package controller

import (
	"testing"

	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/model"
)

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
