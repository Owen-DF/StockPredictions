import pandas as pd




company = "United Rentals"

# Load the dataset
if company == "IBM":
    datasetFULL = pd.read_csv('ExcelFiles/IBM/IBM.csv', index_col='Date', parse_dates=['Date'])
elif company == "United Rentals":
    datasetFULL = pd.read_csv('ExcelFiles/UR/URI.csv', index_col='Date', parse_dates=['Date'])
elif company == "Nvidia":
    datasetFULL = pd.read_csv('ExcelFiles/NVIDIA/NVDA.csv', index_col='Date', parse_dates=['Date'])


datasetEnd = datasetFULL.index.max() - pd.DateOffset(months=3)
print(f"End date excluding the last 3 months: {datasetEnd}")

# Filter the dataset to exclude the last 3 months
dataset_filtered = datasetFULL[datasetFULL.index > datasetEnd]








