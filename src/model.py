# model utilities
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.utils import shuffle

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):
    """Build and compile LSTM model for credit risk prediction."""
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm_model(model, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, validation_split=0.2, verbose=1):
    """Train the LSTM model."""
    if X_val is not None and y_val is not None:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_data=(X_val, y_val), verbose=verbose)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_split=validation_split, verbose=verbose)
    return history

def evaluate_lstm_model(model, X_test, y_test):
    """Evaluate the trained LSTM model."""
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return accuracy, f1, precision, recall

def monte_carlo_evaluation(X, y, selected_features, n_splits=5, epochs=10, batch_size=32):
    """Perform Monte Carlo cross-validation for LSTM model."""
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for _ in range(n_splits):
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
        model = build_lstm_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

        # Train the model
        model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate the model
        y_pred_prob = model.predict(X_test_reshaped)
        y_pred = (y_pred_prob > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Display average metrics
    print("\nAverage Metrics across Monte Carlo splits:")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")

    return {
        'accuracy': np.mean(accuracy_scores),
        'f1': np.mean(f1_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores)
    }

def predict_credit_risk(model, X_new):
    """Make predictions on new data."""
    predictions_probs = model.predict(X_new)
    predictions_binary = (predictions_probs > 0.5).astype(int)
    return predictions_probs, predictions_binary
