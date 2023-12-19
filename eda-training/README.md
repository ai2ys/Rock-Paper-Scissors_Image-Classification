# Information about EDA and Training

**Table of Contents**

1. [Pre-requisites](#pre-requisites)
1. [Environment Setup 🐳](#environment-setup)
    1. [Adapting the `.env` file](#adapting-the-env-file)
    1. [Building the Docker image and running the Docker container](#building-the-docker-image-and-running-the-docker-container)
1. [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
1. [Training](#training)
    1. [Training - Jupyter Notebook 📓](#training---jupyter-notebook)
    1. [Training - Python Script 🐍](#training---python-script)


## Pre-requisites

All of the following code assumes working on `Linux`. Prerequisites are `Docker` and `Docker Compose`, where `Docker Compose` is usually already installed with the latest Docker versions.

```bash
# using docker version
$ docker --version
Docker version 24.0.7, build afdd53b

# using docker compose version
$ docker compose version
Docker Compose version v2.21.0
```

The system has a NVIDIA GPU installed which was used for training the models.

## Environment Setup

A Docker container is used for the development environment. The Docker image is based on `tensorflow/tensorflow:2.14.0-gpu-jupyter` which comes already with `tensorflow-gpu` and other packages installed. Additionally to that the following packages defined in [`docker/requirements.txt`](docker/requirements.txt) are installed in that Docker image.


### Adapting the [`.env`](.env) file

There is a file [`.env`](.env) in the directory of `eda-training`, which you have to adapt to your system editing the following variables.

1. **`DC_PROFILE`** - Set its value to `cpu` or `gpu` depending on your system, if you want to use the GPU for training set it to `gpu`, otherwise to `cpu`. The value `gpu` requires a NVIDIA GPU with CUDA support and the NVIDIA Docker runtime installed.

    ```.env	
    DC_PROFILE=<"cpu" or "gpu">
    ```

1. **`GPU_ID`** - Set its value to the GPU number you want to use for training, if you have multiple NVIDIA GPUs on your system. The value `0` is the default value and will use GPU 0. Instead of the index the `GUID` can be used as well.

    ```.env	
    GPU_ID=<GPU number>
    ```

1. **`PORT_NB`** - Change the port for JupyterLab if required default is `8888`.

    ```.env	
    PORT_NB=<port number>
    ```

### Building the Docker image and running the Docker container 

After adapting the [`.env`](.env) file, you can build the Docker image and run the Docker container. The [`docker-compose.yml`](docker-compose.yml) file contains two profiles, one for `cpu` and one for `gpu`. You can choose the profile by setting the environment variable `DC_PROFILE` to `cpu` or `gpu` before running the commands below.

`Docker Compose` is used as it makes the building process and starting the container easier. For example does the tag name not to have defined explicitly, as it already defined in the [`docker-compose.yml`](docker-compose.yml) file or in the companion environment file [`.env`](.env).

**Important:** In the terminal change to the directory of this README file. Exec

```bash
cd eda-training
```

1. Build the Docker image using `Docker Compose`
    ```bash
    # run "source .env" to make variables known in terminal
    source .env 
    docker compose build eda-training-$DC_PROFILE
    ```

1. Run the Docker container using `Docker Compose`
    ```bash
    # if not the same terminal as previously run "source .env" again
    source .env
    docker compose run --rm --service-ports eda-training-$DC_PROFILE
    ```

    This will start Jupyter Lab in the container and open it in your browser. The terminal will show the URL to open in your browser, which will look like this:

    ```bash
    $ source .env
    $ docker compose run --rm --service-ports eda-training-$DC_PROFILE
    [+] Building 0.0s (0/0)                                                                                                                                                                                                                           
    [+] Building 0.0s (0/0)                                                                                                                                                                                                                           
    [I 2023-12-10 08:35:57.586 ServerApp] Package jupyterlab took 0.0000s to import
    
    ... additional output omitted here ...
    
    [I 2023-12-10 08:35:57.808 ServerApp] Jupyter Server 2.7.3 is running at:
    [I 2023-12-10 08:35:57.808 ServerApp] http://tensorflow-gpu:8888/lab?token=4692485b4dbc9c24134fad54d49e195556f0d69b5e5c30ea
    [I 2023-12-10 08:35:57.808 ServerApp]     http://127.0.0.1:8888/lab?token=4692485b4dbc9c24134fad54d49e195556f0d69b5e5c30ea
    [I 2023-12-10 08:35:57.808 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 2023-12-10 08:35:57.810 ServerApp] 
        
        To access the server, open this file in a browser:
            file:///.local/share/jupyter/runtime/jpserver-1-open.html
        Or copy and paste one of these URLs:
            http://tensorflow-gpu:8888/lab?token=4692485b4dbc9c24134fad54d49e195556f0d69b5e5c30ea
            http://127.0.0.1:8888/lab?token=4692485b4dbc9c24134fad54d49e195556f0d69b5e5c30ea
    ```

## EDA (Exploratory Data Analysis)

After starting the container which will start Juypter Lab, you can open the notebook [`notebook-eda.ipynb`](notebook-eda.ipynb) in Jupyter Lab.

## Training

### Training - Jupyter Notebook

After starting the container which will start Juypter Lab, you can open the notebook [`notebook-training.ipynb`](notebook-training.ipynb) in Jupyter Lab.


### Training - Python Script

The code for training the model has been exported from the notebook [`notebook-training.ipynb`](notebook-training.ipynb) to the Python script [`train.py`](train.py). The Python script can be run from the terminal in the Docker container.

```bash 
# run "source .env" to make variables known in terminal
source .env
docker compose run --rm --service-ports eda-training-$DC_PROFILE python train.py
```

After the training was finished the saved model directory was copied to the project directory one level above this README file. Ensuring that another training run will not overwrite the saved model directory.

```bash	
cp -r models ../
``` 

<!-- Git-LFS support has been added to this repository.

```bash	
# if git-lfs is not installed yet
sudo apt-get update
sudo apt-get install git-lfs
``` -->





