import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

def load_file(file_path):
    global df
    if file_path:
        try:
            # Clean up the file path
            if file_path.startswith("{") and file_path.endswith("}"):
                file_path = file_path[1:-1]
            file_path = file_path.replace("\\", "/")

            # Determine the file format and read accordingly
            if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format")

            status_label.config(text="File loaded successfully.")
            df.head()  # Display first few rows of DataFrame for verification
            train_model()  # Trigger model training after data load
        except Exception as e:
            status_label.config(text=f"Failed to load file: {e}")

def on_drop(event):
    # Handle the file drop event
    file_path = event.data
    load_file(file_path)

def display_data():
    global df
    if df is not None:
        text_widget.delete(1.0, tk.END)  # Clear the text widget
        text_widget.insert(tk.END, df.to_string())  # Insert the DataFrame contents
    else:
        status_label.config(text="No data to display.")

def train_model():
    global df

    if df is not None:
        try:
            # Preprocessing
            training_set = df[:'2016'].iloc[:, 1:2].values
            test_set = df['2017':].iloc[:, 1:2].values
            sc = MinMaxScaler(feature_range=(0, 1))
            training_set_scaled = sc.fit_transform(training_set)
            
            X_train, y_train = [], []
            for i in range(60, len(training_set_scaled)):
                X_train.append(training_set_scaled[i-60:i, 0])
                y_train.append(training_set_scaled[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Model Architecture
            regressorGRU = Sequential()
            regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
            regressorGRU.add(Dropout(0.2))
            regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
            regressorGRU.add(Dropout(0.2))
            regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
            regressorGRU.add(Dropout(0.2))
            regressorGRU.add(GRU(units=50, activation='tanh'))
            regressorGRU.add(Dropout(0.2))
            regressorGRU.add(Dense(units=1))
            regressorGRU.compile(optimizer=SGD(learning_rate=0.01, weight_decay=1e-7, momentum=0.9, nesterov=False),
                                 loss='mean_squared_error')

            # Model Training
            regressorGRU.fit(X_train, y_train, epochs=50, batch_size=150, verbose=2)

            # Predictions
            dataset_total = pd.concat((df["High"][:'2016'], df["High"]['2017':]), axis=0)
            inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values.reshape(-1, 1)
            inputs = sc.transform(inputs)

            X_test = []
            for i in range(60, len(test_set) + 60):
                X_test.append(inputs[i-60:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            predicted_stock_price = regressorGRU.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)

            # Plotting
            plot_predictions(test_set, predicted_stock_price)

        except Exception as e:
            status_label.config(text=f"Failed to train model: {e}")

    else:
        status_label.config(text="No data to train model with.")

def plot_predictions(test, predicted):
    plt.plot(test, color='red', label='Real IBM Stock Price')
    plt.plot(predicted, color='blue', label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

# Initialize the global DataFrame variable
df = None

# Create the main application window
root = TkinterDnD.Tk()

# Set the window title
root.title("Stock Price Prediction App")

# Create a label to indicate drag-and-drop functionality
label = tk.Label(root, text="Drag and drop an Excel or CSV file here", width=50, height=10)
label.pack(pady=20)

# Create a Text widget to display the file data
text_widget = tk.Text(root, wrap=tk.NONE, width=80, height=20)
text_widget.pack(pady=20)

# Create a status label to show messages
status_label = tk.Label(root, text="", fg="red")
status_label.pack(pady=10)

# Create a button to display the DataFrame contents
display_button = tk.Button(root, text="Display Data", command=display_data)
display_button.pack(pady=10)

# Bind the drop event to the on_drop function
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

# Run the Tkinter event loop
root.mainloop()
