#!/bin/bash
# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "docker-compose could not be found. Please install docker-compose."
    exit 1
fi

# Start services with the 'dev' profile
docker-compose --profile dev up