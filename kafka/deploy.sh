kubectl delete daemonset kafka-daemonset
kubectl apply -f kafka-configmap.yaml
kubectl delete daemonset kafka-daemonset -n aide-fl
kubectl apply -f kafka-daemonset.yaml