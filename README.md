# framework

## MLflow utils
- sdf
- qsdf

## Docker workflow

introduction with reason why

### Installation

TODO: add actual docker installation (also add to commands.sh)

```shell
# Make commands script executable
chmod +x docker/commands.sh
```

```shell
# Build containers
docker-compose build
```

Usage example
```shell
# Start MLflow
./scripts/dev_env.sh start

# Create a Python 3.9 dev environment
./scripts/dev_env.sh create 39

# List running containers
./scripts/dev_env.sh list

# Attach to a Python 3.9 container
./scripts/dev_env.sh attach ml_dev_py39
```

Commit docker container to new image
```shell
# While container is running
docker commit ml_dev_py39 my_custom_ml_dev_py39:latest

# Now you can recreate this container anytime
docker run -it my_custom_ml_dev_py39
````

### MLflow container

explain

#### Steps

1.

### Vanilla container (only pip packages)

explain

#### Steps

1. Make sure the MLflow container is running

2. Setup project folder

3. Create vanilla docker container

4. Enter docker container

5. Setup virtual environment

6. Cook

### Special container (special installations required)

explain

#### Steps

1. Make sure the MLflow container is running

2. Setup project folder

3. Create vanilla docker container

4. ??????

4. Enter docker container

5. Setup virtual environment

6. Cook
