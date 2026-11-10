package configuration

import (
	"time"

	"github.com/idlab-discover/AIDE-FL/client/orchestrator/repository"
	"github.com/idlab-discover/AIDE-FL/client/orchestrator/repository/impl"
	"github.com/segmentio/kafka-go"
)

type KafkaFactory interface {
	NewKafkaReaderRepository(topic string) repository.KafkaReaderRepository
	NewKafkaWriterRepository(topic string) repository.KafkaWriterRepository
}

type kafkaFactoryImpl struct {
	Config
}

func NewKafkaFactory(config Config) KafkaFactory {
	return &kafkaFactoryImpl{Config: config}
}

func (factory *kafkaFactoryImpl) NewKafkaReaderRepository(topic string) repository.KafkaReaderRepository {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  []string{factory.Config.Get("KAFKA_ADDRESS")},
		Topic:    topic,
		GroupID:  factory.Config.Get("KAFKA_GROUP"),
		MinBytes: 1,    // 1B
		MaxBytes: 10e6, // 10MB
		MaxWait:  1 * time.Second,
	})

	return impl.NewKafkaReaderRepositoryImpl(reader)
}

func (factory *kafkaFactoryImpl) NewKafkaWriterRepository(topic string) repository.KafkaWriterRepository {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(factory.Config.Get("KAFKA_ADDRESS")),
		Topic:        topic,
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
		BatchSize:    1,
		WriteTimeout: 5 * time.Second,
	}

	return impl.NewKafkaWriterRepositoryImpl(writer)
}
