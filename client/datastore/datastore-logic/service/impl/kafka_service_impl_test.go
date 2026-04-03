package impl

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/exception"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/model"
	"github.com/Str-Gen/AIDE-FL/client/datastore/datastore-logic/service"
	"github.com/segmentio/kafka-go"
)

type stubKafkaRepository struct {
	readMessage      kafka.Message
	committed        []kafka.Message
	preparedMessages []kafka.Message
	generalMessages  []kafka.Message
	failGeneralWrite bool
}

func (repo *stubKafkaRepository) ReadPrepareScenario(context.Context) kafka.Message {
	return repo.readMessage
}

func (repo *stubKafkaRepository) CommitPrepareScenario(_ context.Context, msg kafka.Message) {
	repo.committed = append(repo.committed, msg)
}

func (repo *stubKafkaRepository) WritePreparedScenario(msg kafka.Message) {
	repo.preparedMessages = append(repo.preparedMessages, msg)
}

func (repo *stubKafkaRepository) WriteGeneralMessage(msg kafka.Message) {
	if repo.failGeneralWrite {
		panic(errors.New("write failed"))
	}
	repo.generalMessages = append(repo.generalMessages, msg)
}

func (repo *stubKafkaRepository) Close() {}

func newScenarioMessage(t *testing.T) kafka.Message {
	t.Helper()

	payload, err := json.Marshal(model.Scenario{
		PrepID: "prep-1",
		Name:   "scenario-1",
	})
	if err != nil {
		t.Fatalf("marshal scenario: %v", err)
	}

	return kafka.Message{Key: []byte("prep-1"), Value: payload}
}

func newKafkaServiceForTest(t *testing.T, repo *stubKafkaRepository) *kafkaServiceImpl {
	t.Helper()

	var kafkaSvc service.KafkaService = &kafkaServiceImpl{KafkaRepository: repo}
	exception.RegisterKafkaService(&kafkaSvc)
	return kafkaSvc.(*kafkaServiceImpl)
}

func TestHandlePrepareScenarioCommitsAfterSuccess(t *testing.T) {
	repo := &stubKafkaRepository{readMessage: newScenarioMessage(t)}
	service := newKafkaServiceForTest(t, repo)

	service.handlePrepareScenario(context.Background(), repo.readMessage, func(model.Scenario) error {
		return nil
	})

	if len(repo.committed) != 1 {
		t.Fatalf("expected 1 committed message, got %d", len(repo.committed))
	}
	if len(repo.generalMessages) != 0 {
		t.Fatalf("expected no general messages, got %d", len(repo.generalMessages))
	}
}

func TestHandlePrepareScenarioCommitsAfterReturnedError(t *testing.T) {
	repo := &stubKafkaRepository{readMessage: newScenarioMessage(t)}
	service := newKafkaServiceForTest(t, repo)

	service.handlePrepareScenario(context.Background(), repo.readMessage, func(model.Scenario) error {
		return errors.New("boom")
	})

	if len(repo.committed) != 1 {
		t.Fatalf("expected 1 committed message, got %d", len(repo.committed))
	}
	if len(repo.generalMessages) != 1 {
		t.Fatalf("expected 1 general message, got %d", len(repo.generalMessages))
	}
}

func TestHandlePrepareScenarioCommitsAfterLoggedPanic(t *testing.T) {
	repo := &stubKafkaRepository{readMessage: newScenarioMessage(t)}
	service := newKafkaServiceForTest(t, repo)

	service.handlePrepareScenario(context.Background(), repo.readMessage, func(model.Scenario) error {
		exception.PanicLogging(errors.New("boom"), "Unable to prepare scenario")
		return nil
	})

	if len(repo.committed) != 1 {
		t.Fatalf("expected 1 committed message, got %d", len(repo.committed))
	}
	if len(repo.generalMessages) != 1 {
		t.Fatalf("expected 1 general message, got %d", len(repo.generalMessages))
	}
}

func TestHandlePrepareScenarioDoesNotCommitWhenErrorPublishFails(t *testing.T) {
	repo := &stubKafkaRepository{
		readMessage:      newScenarioMessage(t),
		failGeneralWrite: true,
	}
	service := &kafkaServiceImpl{KafkaRepository: repo}

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when publishing general message fails")
		}
		if len(repo.committed) != 0 {
			t.Fatalf("expected no committed messages, got %d", len(repo.committed))
		}
	}()

	service.handlePrepareScenario(context.Background(), repo.readMessage, func(model.Scenario) error {
		return errors.New("boom")
	})
}
