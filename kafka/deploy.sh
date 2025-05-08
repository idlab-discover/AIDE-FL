kubectl delete daemonset kafka-daemonset
kubectl apply -f kafka-configmap.yaml
kubectl apply -f kafka-daemonset.yaml