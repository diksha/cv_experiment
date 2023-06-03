# States/Events processing POC

This is a POC flink pipeline which retrieves State/Event messages from S3 and runs our incident pipeline on that data. Here's how you can run this yourself:

1. Set up a local kubernetes cluster using either microk8s or minikube with the following requirements
   a. Must have DNS support
   b. Must have a registry (this code assumes microk8s registry which is at localhost:32000, change the `push` target to add your registry address if it is different)
2. Install flink-k8s-operator following the directions here: [https://github.com/spotify/flink-on-k8s-operator/blob/master/docs/user_guide.md]
3. Run the push target to push this image to your registry `bazel run //experimental/jorge/try_flink:push`
4. Create the flink cluster/job `kubectl apply -f cluster.yaml`
5. Watch the logs `kubectl logs -f cluster-taskmanager-0` (you may have to wait a moment for the container to be created)
