#!/bin/bash

: ${PORT:=6011}

COMMAND="jupyter notebook --port $PORT --ip 0.0.0.0 --allow-root"

echo "Launching Jupyter Notebook on port $PORT"
echo "To access the notebook, visit either:"
echo "* http://localhost:$PORT/?token=<token>"
echo "* http://$(hostname):$PORT/?token=<token>"
echo ""
echo "NOTE: jupyter notebook will shortly print a url with port=8888 hardcoded. "
echo "Please replace this with your PORT=$PORT."

# wait 3 seconds so that the above message is visible in the terminal
sleep 3

COMMAND=$COMMAND PORTS="$PORT:$PORT" ./scripts/docker/launch.sh "$@"
