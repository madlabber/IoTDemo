#!/bin/bash

# This script maintains a local copy of the doorcam application
# and starts the app
if [ -f "/mnt/distribution_cache/doorcam/doorbell_camera.py" ]; then
	echo "Pulling latest application code..."
	mkdir ~/doorcam
	cp /mnt/distribution_cache/doorcam/doorbell_camera.py ~/doorcam/doorbell_camera.py
	cp /mnt/distribution_cache/doorcam/start.sh ~/doorcam/start.sh
	echo "Done."
	echo
fi
echo "Starting application..."
python3 ~/doorcam/doorbell_camera.py
echo "Done."
