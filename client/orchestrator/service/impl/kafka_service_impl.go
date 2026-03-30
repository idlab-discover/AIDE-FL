package impl

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/exception"
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/model"
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/repository"
	"github.com/MartienskieNum1/AIDE-FL/client/orchestrator/service"
	"github.com/segmentio/kafka-go"
)

type kafkaServiceImpl struct {
	prepareWriterRepo        repository.KafkaWriterRepository
	preparedReaderRepo       repository.KafkaReaderRepository
	preprocessWriterRepo     repository.KafkaWriterRepository
	preprocessedReaderRepo   repository.KafkaReaderRepository
	trainWriterRepo          repository.KafkaWriterRepository
	trainedReaderRepo        repository.KafkaReaderRepository
	generalMessageReaderRepo repository.KafkaReaderRepository
}

func NewKafkaServiceImpl(
	prepareWriterRepo *repository.KafkaWriterRepository,
	preparedReaderRepo *repository.KafkaReaderRepository,
	preprocessWriterRepo *repository.KafkaWriterRepository,
	preprocessedReaderRepo *repository.KafkaReaderRepository,
	trainWriterRepo *repository.KafkaWriterRepository,
	trainedReaderRepo *repository.KafkaReaderRepository,
	generalMessageReaderRepo *repository.KafkaReaderRepository,
) service.KafkaService {
	return &kafkaServiceImpl{
		prepareWriterRepo:        *prepareWriterRepo,
		preparedReaderRepo:       *preparedReaderRepo,
		preprocessWriterRepo:     *preprocessWriterRepo,
		preprocessedReaderRepo:   *preprocessedReaderRepo,
		trainWriterRepo:          *trainWriterRepo,
		trainedReaderRepo:        *trainedReaderRepo,
		generalMessageReaderRepo: *generalMessageReaderRepo,
	}
}

func (service *kafkaServiceImpl) PublishPrepareScenario(scenario model.Scenario) {
	slog.Info("Publishing prepare", "scenario", scenario.Name, "prepId", scenario.PrepID)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.PrepID),
		Value: byteArray,
	}

	service.prepareWriterRepo.Write(msg)
}

func (service *kafkaServiceImpl) PublishPreprocessScenario(scenario model.Scenario) {
	slog.Info("Publishing preprocess", "scenario", scenario.Name, "prepId", scenario.PrepID)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.PrepID),
		Value: byteArray,
	}

	service.preprocessWriterRepo.Write(msg)
}

func (service *kafkaServiceImpl) PublishTrainScenario(scenario model.Scenario) {
	slog.Info("Publishing train", "scenario", scenario.Name, "prepId", scenario.PrepID)

	byteArray, err := json.Marshal(scenario)
	exception.PanicLogging(err, "Unable to marshal scenario to JSON")

	msg := kafka.Message{
		Key:   []byte(scenario.PrepID),
		Value: byteArray,
	}

	service.trainWriterRepo.Write(msg)
}

func (service *kafkaServiceImpl) ConsumePreparedScenario(ctx context.Context, expectedPrepID string) (model.Scenario, error) {
	return service.consumeScenario(ctx, service.preparedReaderRepo, "prepared", expectedPrepID)
}

func (service *kafkaServiceImpl) ConsumePreprocessedScenario(ctx context.Context, expectedPrepID string) (model.Scenario, error) {
	return service.consumeScenario(ctx, service.preprocessedReaderRepo, "preprocessed", expectedPrepID)
}

func (service *kafkaServiceImpl) ConsumeTrainedScenario(ctx context.Context, expectedPrepID string) (model.Scenario, error) {
	return service.consumeScenario(ctx, service.trainedReaderRepo, "trained", expectedPrepID)
}

func (service *kafkaServiceImpl) ConsumeGeneralMessage(ctx context.Context) (model.GeneralMessage, error) {
	slog.Info("Consuming general messages")

	m, err := service.generalMessageReaderRepo.Read(ctx)
	if err != nil {
		return model.GeneralMessage{}, err
	}

	var message model.GeneralMessage
	err = json.Unmarshal(m.Value, &message)
	exception.PanicLogging(err, "Unable to unmarshal JSON to general message")

	slog.Info("Received general message", "type", message.Type, "origin", message.Origin)

	service.generalMessageReaderRepo.Commit(ctx, m)

	return message, err
}

func (service *kafkaServiceImpl) consumeScenario(ctx context.Context, reader repository.KafkaReaderRepository, stage string, expectedPrepID string) (model.Scenario, error) {
	slog.Info(fmt.Sprintf("Consuming %s scenario messages", stage), "expectedPrepID", expectedPrepID)

	for {
		m, err := reader.Read(ctx)
		if err != nil {
			return model.Scenario{}, err
		}

		var scenario model.Scenario
		err = json.Unmarshal(m.Value, &scenario)
		exception.PanicLogging(err, "Unable to unmarshal JSON to scenario")

		if scenario.PrepID != expectedPrepID {
			slog.Warn(
				fmt.Sprintf("Discarding stale %s scenario", stage),
				"expectedPrepID", expectedPrepID,
				"receivedPrepID", scenario.PrepID,
				"receivedScenario", scenario.Name,
			)
			reader.Commit(ctx, m)
			continue
		}

		slog.Info(fmt.Sprintf("Received %s", stage), "scenario", scenario.Name)
		reader.Commit(ctx, m)

		return scenario, nil
	}
}
