# Image Classification - Rock Paper Scissors Game

**Table of contents**

1. [General information](#general-information)
    1. [General information on the game](#general-information-on-the-game)
    1. [Goal of this project](#goal-of-this-project)
    1. [Dataset](#dataset)
1. [EDA and Model Training](#eda-and-model-training)
    1. [Environment setup using Docker container](#environment-setup-using-docker-container)
    1. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    1. [Model Training](#model-training)
        1. [Training - Jupyter Notebook 📓](#training---jupyter-notebook)
        1. [Training - Python Script 🐍](#training---python-script)
1. [Deployment](#deployment)
    1. [Model Serving - Docker Compose](#model-serving---docker-compose)
    1. [Model Serving - Kubernetes](#model-serving---kubernetes)
        1. [Installing `kubectl` and `kind` on Linux](#installing-kubectl-and-kind-on-linux)
        1. [Creating a Kubernetes cluster using `kind` and `kubectl`](#creating-a-kubernetes-cluster-using-kind-and-kubectl)
        1. [Trouble Shooting using Docker rootless mode](#trouble-shooting-using-docker-rootless-mode)
1. [MLZoomCamp Capstone Project 1 General Information](#mlzoomcamp-capstone-project-1-general-information)


## General information

This repository contains the code for the MLZoomCamp Capstone Project 1.

### General information on the game

🎲 [ Game 'rock paper scissors' described on Wikipedia](https://en.wikipedia.org/wiki/Rock_paper_scissors)

### Goal of this project

The goal of this project is to automatically classify images of hands showing gestures of the game "Rock Paper Scissors" as 'rock', 'paper', or 'scissors' gesture.

The model training is done using the `TensorFlow` Machine Learning framework. 

The trained model will then get deployed using `Kubernetes` (locally using `kind`).

In `Kubernetes` the model will be served as a pod using `TensorFlow Serving` (`RESTful API`) and using a gateway service (`REST` API using `Flask`).

- Kubernetes
    - https://kubernetes.io/
    
- kind
    - https://kubernetes.io/docs/tasks/tools/#kind
     
- TensorFlow Serving
    - https://www.tensorflow.org/tfx/guide/serving

- Flask
    - https://flask.palletsprojects.com/en/3.0.x/

The model was trained and deployed using a `Linux` system with a `NVIDIA` GPU installed.  EDA and training were executed in Docker containers. The system has Docker (rootless mode) installed. Therefore, some adaptions had to be made to be able to use `kind`, see section [Trouble Shooting using Docker rootless mode](#trouble-shooting-using-docker-rootless-mode).


### Dataset

The dataset used in this project is available in the **TensorFlow datasets catalog**. 

- [https://www.tensorflow.org/datasets/catalog/rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)

- The dataset is not included in this repository. 

- An internet connection is required to download the 
dataset using the Python packages `tensorflow-datasets`.

The following example images were taken from this dataset. It can be seen that the dataset contains hands in various skin tones capturing wide variety of hand shapes and orientations. The images are `300x300` pixels in size and are color images (RGB). 
The images seem all to have a bright background and the hands are in the center of the images. 
Furthermore the hands seem always to be fully visible in the images, but seem to have different scales. 

![rock_paper_scissors_examples.png](rock_paper_scissors_examples.png)

The dataset itself only contains `test` and `train` splits. A `validation` split will be created taking a proportion of images from the `train` split. 

```text
Number of training (full train) examples: 2520
Number of test examples                 : 372
Ratio test data to (full) train data    : 14.76%
```

When comparing the amount of test data to the amount of training (here referred to as 'full train') data we can see that it is approximately 15% of the training data. 
Therefore the `train` split is further split into `train` and `validation` splits using 15% of the `train` split for creating the `validation` split.

## EDA and Model Training 

Content covered in [./eda-training/README.md](./eda-training/README.md) 

1. Environment setup using Docker container
1. Exploratory Data Analysis (EDA)
1. Model Training

### Environment setup using Docker container
Please refer to the notebook [./eda-training/README.md](./eda-training/README.md) for how to set up the development environment for running the EDA and Model Training.



## Deployment

We are going to use `TensorFlow Serving` using its `RESTful API` to deploy the model.

Following Docker images will be used for deployment:

- [`model.dockerfile`](./model.dockerfile) - Docker image for serving the model using TensorFlow Serving
    - No requirements file is needed as the image is based on the official TensorFlow Serving image
- [`gateway.dockerfile`](./gateway.dockerfile) - Docker image for the gateway which will be used to send requests to the model server
    - **Environment setup**: Requirements file [`gateway-requirements.txt`](./gateway-requirements.txt) is used to install the required Python packages in the Docker image

### Model Serving - Docker Compose

Prior to the deployment to a Kubernetes cluster we are going to test the model and the gateway locally using `Docker Compose`. Therefore, we are going to use the `Docker Compose` file [`docker-compose.yml`](docker-compose.yml) which defines the services `tf-serving` and `gateway`.
Additionally, a service `test` is defined which is used for testing the model and the gateway locally.

The docker images are built using `docker compose` and pushed to Docker Hub. Therefore, the build step could be omitted. Directly running the `docker compose up` (using the `--profile test`) command below will pull all the images from Docker Hub.

Docker images are provided on Docker Hub:

- https://hub.docker.com/repository/docker/ai2ys/mlzoomcamp-capstone-1
    - `ai2ys/mlzoomcamp-capstone-1:test-model-0.0.0`
    - `ai2ys/mlzoomcamp-capstone-1:tf-serving-0.0.0`
    - ` ai2ys/mlzoomcamp-capstone-1:gateway-0.0.0`

Open a terminal and make sure to be in the directory of this README file and run the following commands to test the serving and gateway locally using `Docker Compose`.

```bash
# [optional] build the Docker image, pre-built images are provided on Docker Hub
docker compose --profile test build 
# start the services (containers) using the 'test' profile
docker compose --profile test up -d 
# test tf-serving
docker exec -it test python test-model.py
# test gateway calling tf-serving
docker exec -it test python test-gateway.py
# stop and remove the containers
docker compose --profile test down
```

After knowing that the model and the gateway are working locally we can deploy the model and the gateway to a `Kubernetes` cluster.

Before doing that we are going to push the Docker images to a Docker registry.

```bash	
# login to Docker Hub
docker login
# ... login omitted here (user, password, ...)

# push the images to Docker Hub
docker compose push
```

### Model Serving - Kubernetes

For deploying the model (pods) and the gateway (services) to a `Kubernetes` cluster we are going to use the tool `kind` (Kubernetes in Docker) to create a `Kubernetes` cluster locally.

This step assumes that the `Docker` images have already been built locally or pulled from Docker Hub, see section [Model Serving - Docker Compose](#model-serving---docker-compose).

#### Installing `kubectl` and `kind` on Linux

Installing `kubectl` and `kind` on Linux (AMD64 / x86_64).

1. Install `kubectl` on Linux (AMD64 / x86_64):<br>
https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/

    ```bash	
    # download latest realease
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

    # download checksum
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"

    # validate checksum
    echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check

    # install kubectl
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

    # ensure installed version is up-to-date
    kubectl version --client
    ```	

1. Install `kind` on Linux (AMD64 / x86_64):<br>
https://kind.sigs.k8s.io/docs/user/quick-start/#installing-from-release-binaries

    ```bash
    # intall from release binaries
    [ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
    kind --version
    ```

#### Creating a Kubernetes cluster using `kind` and `kubectl`

🎞️ **Video recording of performing the following steps**: 🔗 https://youtu.be/AefPENIT-jA

1. Create a Kubernetes cluster using `kind` and `kubectl`:

    ```bash
    # create a cluster
    kind create cluster --name mlzoomcamp-capstone-1
    # check context of kubectl
    kubectl config current-context
    # check cluster info
    kubectl cluster-info

    # load docker images into the named cluster
    kind load docker-image \
        ai2ys/mlzoomcamp-capstone-1:tf-serving-0.0.0 \
        ai2ys/mlzoomcamp-capstone-1:gateway-0.0.0 \
        --name mlzoomcamp-capstone-1

    # create/update resources
    kubectl apply -f k8s/tf-serving-deployment.yaml
    kubectl apply -f k8s/tf-serving-service.yaml
    kubectl apply -f k8s/gateway-deployment.yaml
    kubectl apply -f k8s/gateway-service.yaml

    kubectl get pod
    kubectl get services

    # forwarding both ports to localhost
    # adding '&' to first command for being able to run second command from same terminal
    kubectl port-forward service/tf-serving 8501:8501 &
    kubectl port-forward service/gateway 9696:9696    
    ```

1. In local terminal run the following command to test the gateway calling the model server:

    ```bash
    python test-gateway.py
    ```

1. In local terminal from the 1st step run command for deleting the cluster.
    ```bash
    # Use CTRL+C to stop the port forwarding

    # delete the cluster
    kind delete cluster --name mlzoomcamp-capstone-1
    ```

#### Trouble Shooting using Docker rootless mode

When creating the cluster using `kind` I got the following error message, because of the system setup with Docker rootless mode:

```bash
ERROR: failed to create cluster: running kind with rootless provider requires cgroup v2, see https://kind.sigs.k8s.io/docs/user/rootless/
```

Following the instructions provided in the link of the error message to fix this issue: https://kind.sigs.k8s.io/docs/user/rootless/

Steps accomplished:

1. > [...] adding `GRUB_CMDLINE_LINUX="systemd.unified_cgroup_hierarchy=1"` to `/etc/default/grub` and running `sudo update-grub` to enable cgroup v2 [...]
    1. Created a backup of `/etc/default/grub`
        ```bash
        sudo cp /etc/default/grub /etc/default/grub.bak
        ```
    
    1. Added `GRUB_CMDLINE_LINUX="systemd.unified_cgroup_hierarchy=1"` to `/etc/default/grub`
        ```bash
        sudo vim /etc/default/grub
        # add mentioned content to file
        ```

1. Rebooting the system 
    ```bash
    sudo reboot now
    ```

1. Create `/etc/systemd/system/user@.service.d/delegate.conf` with the following content, and then run `sudo systemctl daemon-reload`:
    ```text
    [Service]
    Delegate=yes
    ```

    1. Creating the file and adding the content:
        ```bash
        # create directory
        sudo mkdir -p /etc/systemd/system/user@.service.d
        # (create and) edit file
        sudo vim /etc/systemd/system/user@.service.d/delegate.conf
        ```

        - Adding the following lines to the file
            ```text
            [Service]
            Delegate=yes
            ```
    
    1. Running the following command to reload the daemon:
        ```bash
        sudo systemctl daemon-reload
        systemctl --user restart docker
        ```

After that it was possible to create the cluster using `kind create cluster` with Docker rootless mode.



## MLZoomCamp Capstone Project 1 General Information
General information about he MLZoomCamp Midterm Project can be found here: https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects#capstone-1

Information for cohort 2023 can be found here: https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2023/projects.md#capstone-1


