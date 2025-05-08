kubectl run kafka-consumer1 -n aide-fl -ti --image=strimzi/kafka:0.18.0-kafka-2.4.0 --restart=Never --overrides='{
  "apiVersion": "v1",
  "spec": {
    "hostNetwork": true,
    "nodeSelector": {
      "kubernetes.io/hostname": "384731d7-fa3f-460d-ab57-md-0-zdjrl-jzkmm"
    }
  }
}' -- bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic scenario.prepared --from-beginning