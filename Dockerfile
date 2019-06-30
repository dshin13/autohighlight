# Download base image from AWS
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04

# Install dependencies
RUN pip install moviepy

# Add autohighlight code and move to directory
COPY models ./models
COPY utils ./utils
COPY autohighlight.py ./autohighlight.py

# Set working directory

