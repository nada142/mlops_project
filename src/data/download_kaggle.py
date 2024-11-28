import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset to download and output path
dataset_identifier = "saldenisov/recipenlg"  # Replace with your dataset
output_path = "data/raw"

# Download and unzip the dataset
api.dataset_download_files(dataset_identifier, path=output_path, unzip=True)
