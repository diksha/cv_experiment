# DataPipeline - BOWSER

## RoadMap

for more information, see <https://app.clickup.com/36001183/v/l/6-901001215165-1>

## Goal

The Bowser datapipeline have so far 4 goals :

- Replay on demand cameras data
- Be in the Test validation for new feature
- Replace production incident data computing
- Replace FireHose (S3 file generation)

## Usage

### Engine

In order to create your own Bowser processor, you will find basic abstraction classes and pre-made data transformation (flatmaps and reducer )

#### Reducer

Reducer are classes that aggregate or reduce a key to create dimension field like a count or an sum for instance

##### keys

This Class is using the AggregateFunction from bowser engine. It's working with Accumulator which are close to be States
That's means all accumulator used by Bowser will be shared among all Task Manager that actually process this Reducer Manipulation

#### FlatMaps

##### AwsBucketToS3ObjectFlatMap

List and Open each file containing in a Config Bucket from AWS S3

##### AwsS3ObjectToStatesEventsFlatMap

Transform AWS S3Object to a protobuffed Voxel State or Event

##### StateOrEventToIncidentFlatMap

Transform State or Event to Incident by calling the Incident Node from Core Voxel

### Config

#### Proto

Bowser Processor is using a Protobuff schema right before than a Bowser Processor `run_job` method is called.
The `_initialize` method inside the common class are validating the proto.

you will find the proto schema on [here](../../../protos/platform/bowser/v1)

### Tests

## Deployment

You have 3 different way to use bowser

### Local Single thread

On this configuration, you will only have to run your Bowser code via a `bazel` command

in the [Bazel processor file](processors/BUILD.bazel), add a new build

```text
bowser_processor(
    name = "{YOUR_BUILD_NAME}",
    main_src = "{YOUR_PROCESSOR_PYTON}",
)
```

```shell
bazel run //services/platform/bowser:{YOUR_BUILD_NAME}
```

once lunched, you should see Bazel log and Bowser Engine beeing trigger, the program will stop when all the data has been processed

### Local Minikube + Bowser Kubernetes Operator

#### Docker + Registry

Before to be able to execute a Bowser processor inside a Kubernetes Bowser Operator, you will have to install it on your dev box.
You will need of :

- Docker
- Curl
- Bazel

Docker is needed to host the minikube kubernetes platform and also host your own registry
In order to create a Registry you gonna have to do :

1. : Stop Docker
2. : Add `127.0.0.1 docker.local` on your `/etc/hosts` devbox file
3. : Add this JSON `{"insecure-registries" : ["http://docke.local:5000"]}` inside `/etc/docker/daemon.json`. If the file don't exist yet , create it.
4. : Start Docker
5. : Run this command `docker run -e REGISTRY_STORAGE_DELETE_ENABLED=true -d -p 5000:5000 --restart=always --name registry registry:2` to create your registry
6. : Run `docker info` you must see something like this

   ```text
    .....
    Insecure Registries:
     docker.local:5000
    ......
   ```

7. Push your Bowser Processor code docker image in your brand new Docker registry with bazel command `bazel run //services/platform/bowser:push_minikube_local`
8. You know that your setup is up to date if you see something when you enter `curl -X GET http://docker.local:5000/v2/services/platform/bowser/tags/list` in your term.
9. You also know that you setup is ok if you can pull the docker image from the registry by doing `docker pull docker.local:5000/services/platform/bowser:latest`

#### Minikube + Bowser Kube Operator

You will need of :

- Minikube
- kubectl
- Helm

1. Enter `ip a` on your term to display your local ip address here mine is `192.168.58.1`

   ```text
   334: br-5366e1d48d6c: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default
       link/ether 02:42:66:05:73:29 brd ff:ff:ff:ff:ff:ff
       inet 192.168.58.1/24 brd 192.168.58.255 scope global br-5366e1d48d6c
          valid_lft forever preferred_lft forever
       inet6 fe80::42:66ff:fe05:7329/64 scope link
          valid_lft forever preferred_lft forever

   ```

2. Create or update `~/.minikube/files/etc/hosts` and add it your ip address found in 1 then docker.local example

   ```text
   192.168.58.1    docker.local
   ```

3. If you already have a minikube running you will have to `delete` it first and recreate it with this command
   `minikube start --mount-string="${YOUR_HOME_DIRECTORY_PATH}/.aws:/root/.aws" --mount --memory 8192 --cpus 4 --kubernetes-version=v1.25.3 --insecure-registry="docker.local:5000"`
4. Run this bunch of command line

   ```shell
   #add Bowser Certificate
   kubectl create -f https://github.com/jetstack/cert-manager/releases/download/v1.8.2/cert-manager.yaml
   #Add Bowser Kube Repo
   helm repo add bowser-operator-repo https://downloads.apache.org/flink/flink-kubernetes-operator- <OPERATOR-VERSION >/
   #Install the bowser kubernetes Operator
   helm install bowser-kubernetes-operator bowser-operator-repo/bowser-kubernetes-operator
   #Create a service account used by bowser only
   kubectl create serviceaccount bowser-testing
   #Migrate the default priviledge from default service account to your new service account
   kubectl create clusterrolebinding bowser-role-binding-bowser --clusterrole=edit --serviceaccount=default:bowser-testing
   ```

5. : Open a new term and then enter `minikube dashboard`. You should see something like that

   ```text
   🎉  Opening http://127.0.0.1:36263/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/ in your default browser...
   👉  http://127.0.0.1:36263/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/
   ```

6. From your laptop ( not dev box ) enter this command `ssh -N -L {PORT}:127.0.0.1:{PORT} -L 8081:127.0.0.1:8081 {YOUR_NAME}@{YOUR_NAME}.devbox.voxelplatform.com`. {PORT} here is 36263
7. Open a new tab on your browser and paste this URL `http://127.0.0.1:{PORT}/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/` {PORT} here is 36263
8. You should now be able to see you Minikube dashboard hosted on your dev box minikube broadcast by ssh tunel

#### Deploy your Bowser Processor

1. Log on AWS with `aws sso login` if not already done.
2. `bazel run //services/platform/bowser:push_minikube_local`. should push your new code into the local registry
3. Deploy your bowser kubernetes file with `kubectl create -f {PATH_TO_KUBERNETES_FILE}` or `kubectl rollout restart -n default deployment ${NAME_OF_YOUR_DEPLOYMENT}`
4. You can check the log like this `kubectl logs -f deployment/${NAME_OF_YOUR_DEPLOYMENT}`
5. Open a new Term and type `kubectl port-forward svc/{NAME_OF_YOUR_DEPLOYMENT}-rest 8081`
6. On your laptop ( and not dev-box ) open a new tab and enter `http://localhost:8081/`. That should open your Bowser Dashboard when on Running job you should see it.

#### Delete your minikube Bowser

1. `minikube profile list` on your term
2. You should see all your minikube profile
3. `minikube -P bowser delete` to only delete the bowser minikube

### Cloud + Bowser Kubernetes Operator

Comming soon
