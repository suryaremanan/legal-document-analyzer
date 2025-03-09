#!/bin/bash

# Install Python dev tools for distutils
apt-get update
apt-get install -y python3-dev python3-distutils

# Ensure pip is latest
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt 