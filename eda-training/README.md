# Information about EDA and Training

All of the following code assumes working on Linux or WSL2. Prerequisites are `Docker` and `Docker Compose` (usually is already installed when installing `Docker`).

In the terminal change to the directory of this README file.

```bash
cd eda-training
```

## Adapting the [`.env`](.env) file

There is a file [`.env`](.env) in the directory of `eda-training`, which you have to adapt to your system editing the following variables.

<!-- 1. **`USER_IDS`** - Set its value to the output of the following line, in my case it is `1000:1000`

    ```bash
    echo "USER_IDS=$(id -u):$(id -g)"
    ``` -->

1. **`DC_PROFILE`** - Set its value to `cpu` or `gpu` depending on your system, if you want to use the GPU for training set it to `gpu`, otherwise to `cpu`. The value `gpu` requires a NVIDIA GPU with CUDA support and the NVIDIA Docker runtime installed.

    ```.env	
    DC_PROFILE=<"cpu" or "gpu">
    ```

## Building the Docker image and running the Docker container

After adapting the [`.env`](.env) file, you can build the Docker image and run the Docker container. The [`docker-compose.yml`](docker-compose.yml) file contains two profiles, one for `cpu` and one for `gpu`. You can choose the profile by setting the environment variable `DC_PROFILE` to `cpu` or `gpu` before running the commands below. 

1. Build the Docker image using `Docker Compose`
    ```bash
    docker compose --profile $DC_PROFILE build
    ```

1. Run the Docker container using `Docker Compose`
    ```bash
    docker compose --profile $DC_PROFILE up
    ```
    This will start Jupyter Lab in the container and open it in your browser. The terminal will show the URL to open in your browser, which will look like this:

    ```bash
    $ docker compose --profile $DC_PROFILE run --rm --service-ports eda-training-$DC_PROFILE
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
1. Removing the container after stopping it using `Ctrl+C`
    ```bash
    docker compose --profile $DC_PROFILE down
    ```

## EDA (Exploratory Data Analysis)

After starting the container which will start Juypter Lab, you can open the notebook [`notebook-eda.ipynb`](notebook-eda.ipynb) in Jupyter Lab.


## Training

After starting the container which will start Juypter Lab, you can open the notebook [`notebook-training.ipynb`](notebook-training.ipynb) in Jupyter Lab.