##!/bin/bash

# Prepend commands with "microk8s" if needed

helm3 repo add strimzi https://strimzi.io/charts/

helm3 install kafka-operator strimzi/strimzi-kafka-operator

kubectl apply -f - <<EOF
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: kafka-cluster
spec:
  kafka:
    replicas: 1
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
    storage:
      type: ephemeral
  zookeeper:
    replicas: 1
    storage:
      type: ephemeral
  entityOperator:
    topicOperator: {}
    userOperator: {}
EOF

kubectl apply -f - <<EOF
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaTopic
metadata:
  name: "scenario.status"
  labels:
    strimzi.io/cluster: kafka-cluster
spec:
  partitions: 1
  replicas: 1
EOF