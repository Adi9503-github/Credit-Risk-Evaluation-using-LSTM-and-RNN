import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Load data (assuming data is preprocessed as in the notebook)
# For simplicity, using sample data; in real scenario, load from CSV
# Here, we'll simulate the data structure

# Assuming selected_features_optimized is defined; for demo, use some features
selected_features_optimized = ['Credit amount', 'Duration', 'Age']  # Example

# Simulate data
np.random.seed(42)
data = pd.DataFrame({
    'Credit amount': np.random.rand(1000) * 10000,
    'Duration': np.random.randint(1, 72, 1000),
    'Age': np.random.randint(18, 80, 1000),
    'credit_risk': np.random.randint(0, 2, 1000)
})

X = data[selected_features_optimized]
y = data['credit_risk']

n_splits = 1  # For demo, use 1 split to generate images quickly

for split in range(n_splits):
    X_shuffled, y_shuffled = shuffle(X, y, random_state=np.random.randint(1000))
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=np.random.randint(1000))

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the data for LSTM
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Lists to store metrics for each epoch
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []

    # Train the model and collect metrics
    for epoch in range(10):
        history = model.fit(X_train_reshaped, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=0)

        # Calculate on training set
        y_train_pred = (model.predict(X_train_reshaped, verbose=0) > 0.5).astype(int)
        train_precision.append(precision_score(y_train, y_train_pred, zero_division=0))
        train_recall.append(recall_score(y_train, y_train_pred, zero_division=0))
        train_f1.append(f1_score(y_train, y_train_pred, zero_division=0))

        # Calculate on validation set (using test as val for simplicity)
        y_val_pred = (model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int)
        val_precision.append(precision_score(y_test, y_val_pred, zero_division=0))
        val_recall.append(recall_score(y_test, y_val_pred, zero_division=0))
        val_f1.append(f1_score(y_test, y_val_pred, zero_division=0))

    # Plot Precision
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), train_precision, label='Training Precision', color='blue')
    plt.plot(range(1, 11), val_precision, label='Validation Precision', color='orange')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('images/precision_plot.png')
    plt.close()

    # Plot Recall
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), train_recall, label='Training Recall', color='blue')
    plt.plot(range(1, 11), val_recall, label='Validation Recall', color='orange')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig('images/recall_plot.png')
    plt.close()

    # Plot F1 Score
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), train_f1, label='Training F1 Score', color='blue')
    plt.plot(range(1, 11), val_f1, label='Validation F1 Score', color='orange')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('images/f1_score_plot.png')
    plt.close()

print("Metrics images generated successfully!")
