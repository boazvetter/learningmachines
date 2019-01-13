#!/bin/bash

set -e

HOST_IP="$(ip a | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | grep -v '172.*')"

PROJECT_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/"
exec docker run -e "HOST_IP=$HOST_IP" --rm -it -v "${PROJECT_FOLDER}:/root/projects/" cigroup/learning-machines bash

