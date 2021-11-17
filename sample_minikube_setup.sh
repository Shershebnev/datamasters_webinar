# DISCLAIMER: tested on MacBook Pro (16-inch, 2019) with MacOS 10.15.6 installed
MODEL_NAME="resnet18"

minikube start --kubernetes-version=v1.21.4

# kubernetes dashboard
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.4.0/aio/deploy/recommended.yaml

# following https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md

kubectl apply -f kube_server_files/dashboard_adminuser.yaml
kubectl apply -f kube_server_files/admin_role_binding.yaml

# get token for dashboard login
# kubectl -n kubernetes-dashboard get secret $(kubectl -n kubernetes-dashboard get sa/admin-user -o jsonpath="{.secrets[0].name}") -o go-template="{{.data.token | base64decode}}"
# start in separate window
# kubectl proxy

# seldon setup
kubectl create namespace seldon
kubectl config set-context $(kubectl config current-context) --namespace=seldon
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator --repo https://storage.googleapis.com/seldon-charts \
    --set ambassador.enabled=true \
    --set usageMetrics.enabled=true \
    --namespace seldon-system

kubectl rollout status deploy/seldon-controller-manager -n seldon-system

helm repo add datawire https://www.getambassador.io
helm repo update
helm install ambassador datawire/ambassador \
    --set image.repository=docker.io/datawire/ambassador \
    --set crds.keep=false \
    --set enableAES=false \
    --namespace seldon-system

kubectl rollout status deploy/ambassador -n seldon-system

kubectl create ns minio
helm repo add minio https://helm.min.io/
helm install minio minio/minio \
    --set accessKey=admin \
    --set secretKey=password \
    --namespace minio

# if gui for minio needed
# export POD_NAME=$(kubectl get pods --namespace minio -l "release=minio" -o jsonpath="{.items[0].metadata.name}")
# kubectl port-forward $POD_NAME 9000 --namespace minio &

mc alias set minio http://localhost:9000 "admin" "password" --api s3v4
mc ls minio  # test
mc mb minio/models
# cp model directory
mc cp --recursive models/$MODEL_NAME minio/models/animal-model/


kubectl apply -f kube_server_files/seldon_rclone.yaml

kubectl apply -f kube_dev_files/model_deploy.yaml

# run port-forwarding in separate window
kubectl port-forward $(kubectl get pods -n seldon-system -l app.kubernetes.io/name=ambassador -o jsonpath='{.items[0].metadata.name}') -n seldon-system 8003:8080

curl -s http://localhost:8003/seldon/seldon-system/$MODEL_NAME/v2/models/$MODEL_NAME | jq .


# Monitoring
helm install seldon-core-analytics seldon-core-analytics \
   --repo https://storage.googleapis.com/seldon-charts \
   --namespace seldon-system
kubectl rollout status deploy/seldon-core-analytics-grafana -n seldon-system

# run port-forwarding in separate window
kubectl port-forward svc/seldon-core-analytics-grafana 3000:80 -n seldon-system

# web browser: http://localhost:3000