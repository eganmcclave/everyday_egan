#!/bin/bash

echo "Starting Python Pipeline"

python3 main.py -p "photos/original_photos" -f "code/face_details.json" -c -v

echo "Pipeline completed"
