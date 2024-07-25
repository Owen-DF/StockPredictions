import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import logging 

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set TensorFlow environment variable

# Define paths
COMPANY_PATHS = {
    "IBM": {"model": 'ExcelFiles/IBM/IBM.h5', "data": 'ExcelFiles/IBM/IBM.csv'},
    "United Rentals": {"model": 'ExcelFiles/UR/URI.h5', "data": 'ExcelFiles/UR/URI.csv'},
    "Nvidia": {"model": 'ExcelFiles/NVIDIA/Nvidia.h5', "data": 'ExcelFiles/NVIDIA/NVDA.csv'}
}

# Define the company
company = "IBM"

# Function to load dataset and model
def loadDatasetAndModel(company):
    if company not in COMPANY_PATHS:
        raise ValueError("Company not supported.")
    
    model_path = COMPANY_PATHS[company]["model"]
    dataset_path = COMPANY_PATHS[company]["data"]

    try:
        model = load_model(model_path)
        dataset = pd.read_csv(dataset_path, index_col='Date', parse_dates=['Date'])
        return model, dataset
    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
    except ValueError as e:
        logging.error(f"Invalid company or dataset: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

# Load model and dataset
model, dataset = loadDatasetAndModel(company)

# Calculate the beginning date for the test (3 months back)
datasetBeg = dataset.index.max() - pd.DateOffset(months=3)
logging.info(f"Beginning Date for test: {datasetBeg}")

# Filter the dataset to exclude the last 3 months
SIMU_Dates = dataset[dataset.index > datasetBeg]

# Placeholder for predictions
predictions = []

# Further processing and prediction logic would go here...

