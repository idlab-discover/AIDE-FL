package impl

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/Str-Gen/AIDE-FL/client/orchestrator/model"
	"github.com/segmentio/kafka-go"
)

type stubReaderRepo struct {
	messages  []kafka.Message
	readErr   error
	committed []kafka.Message
	readCalls int
	commitErr error
}

func (repo *stubReaderRepo) Read(ctx context.Context) (kafka.Message, error) {
	if repo.readErr != nil {
		return kafka.Message{}, repo.readErr
	}
	if repo.readCalls >= len(repo.messages) {
		<-ctx.Done()
		return kafka.Message{}, ctx.Err()
	}

	message := repo.messages[repo.readCalls]
	repo.readCalls++
	return message, nil
}

func (repo *stubReaderRepo) Commit(context.Context, kafka.Message) {
	repo.committed = append(repo.committed, repo.messages[len(repo.committed)])
}

func (repo *stubReaderRepo) Close() {}

type stubWriterRepo struct{}

func (repo *stubWriterRepo) Write(kafka.Message) {}

func (repo *stubWriterRepo) Close() {}

func TestConsumePreparedScenarioSkipsStaleMessages(t *testing.T) {
	t.Helper()

	service := kafkaServiceImpl{
		preparedReaderRepo: &stubReaderRepo{
			messages: []kafka.Message{
				mustScenarioMessage(t, "stale-prep", "stale-run"),
				mustScenarioMessage(t, "expected-prep", "expected-run"),
			},
		},
	}

	scenario, err := service.ConsumePreparedScenario(context.Background(), "expected-prep")
	if err != nil {
		t.Fatalf("ConsumePreparedScenario returned error: %v", err)
	}

	if scenario.Name != "expected-run" {
		t.Fatalf("ConsumePreparedScenario returned scenario %q, want %q", scenario.Name, "expected-run")
	}
	if scenario.PrepID != "expected-prep" {
		t.Fatalf("ConsumePreparedScenario returned prep id %q, want %q", scenario.PrepID, "expected-prep")
	}

	reader := service.preparedReaderRepo.(*stubReaderRepo)
	if len(reader.committed) != 2 {
		t.Fatalf("committed %d messages, want 2", len(reader.committed))
	}
}

func TestConsumePreparedScenarioHonorsContextCancellation(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	service := kafkaServiceImpl{
		preparedReaderRepo: &stubReaderRepo{},
	}

	_, err := service.ConsumePreparedScenario(ctx, "expected-prep")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("ConsumePreparedScenario error = %v, want %v", err, context.Canceled)
	}
}

func mustScenarioMessage(t *testing.T, prepID string, name string) kafka.Message {
	t.Helper()

	scenario := model.Scenario{PrepID: prepID, Name: name}
	payload, err := json.Marshal(scenario)
	if err != nil {
		t.Fatalf("json.Marshal(%q): %v", name, err)
	}

	return kafka.Message{
		Key:   []byte(name),
		Value: payload,
	}
}
