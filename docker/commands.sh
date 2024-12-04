#!/bin/bash

# Development Environment Management Script

# Ensure script is executable
# Run: chmod +x dev_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Display usage information
usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "$0 {start|stop|create|list|attach|destroy} [options]"
    echo
    echo -e "${GREEN}Commands:${NC}"
    echo "  start                  Start MLflow tracking server"
    echo "  stop                   Stop all containers"
    echo "  create <py_version>    Create a new dev container (37, 38, 39, 310, 311, 312)"
    echo "  list                   List all containers"
    echo "  attach <container>     Attach to a running container"
    echo "  destroy                Remove all containers and volumes"
    exit 1
}

# Start MLflow tracking
start_mlflow() {
    echo -e "${GREEN}Starting MLflow tracking server...${NC}"
    docker compose up -d mlflow
}

# Stop all containers
stop_containers() {
    echo -e "${YELLOW}Stopping all containers...${NC}"
    docker compose down
}

# Create a development container
create_dev_env() {
    local python_version=$1
    if [[ ! $python_version =~ ^(37|38|39|310|311|312)$ ]]; then
        echo -e "${YELLOW}Invalid Python version. Use 37, 38, 39, 310, 311, or 312.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Creating Python $python_version development container...${NC}"
    docker compose up -d ml_dev_py$python_version
}

# List running containers
list_containers() {
    echo -e "${GREEN}Running Containers:${NC}"
    docker ps
}

# Attach to a specific container
attach_container() {
    local container_name=$1
    if [ -z "$container_name" ]; then
        echo -e "${YELLOW}Please specify a container name.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Attaching to container: $container_name${NC}"
    docker exec -it $container_name bash
}

# Destroy all containers and volumes
destroy_environment() {
    echo -e "${YELLOW}WARNING: This will remove ALL containers and volumes!${NC}"
    read -p "Are you sure? (y/N) " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        docker compose down -v
        echo -e "${GREEN}All containers and volumes have been removed.${NC}"
    else
        echo -e "${YELLOW}Operation cancelled.${NC}"
    fi
}

# Main script logic
case "$1" in
    start)
        start_mlflow
        ;;
    stop)
        stop_containers
        ;;
    create)
        create_dev_env $2
        ;;
    list)
        list_containers
        ;;
    attach)
        attach_container $2
        ;;
    destroy)
        destroy_environment
        ;;
    *)
        usage
        ;;
esac
