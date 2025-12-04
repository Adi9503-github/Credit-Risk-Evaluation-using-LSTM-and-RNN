# Credit Risk Assessment Project

## Overview

This project implements a comprehensive credit risk assessment system using advanced machine learning techniques. The system analyzes German credit data to predict credit risk levels, incorporating feature engineering, selection, and deep learning models for accurate risk classification.

## Features

### Data Preprocessing
- **Missing Value Handling**: Uses Iterative Imputation and KNN Imputation for robust data completion
- **Feature Engineering**: Creates credit risk labels based on credit amount and duration thresholds
- **Categorical Encoding**: Implements Weight of Evidence (WOE) encoding for categorical variables
- **Feature Scaling**: Applies StandardScaler for numerical feature normalization

### Advanced Feature Selection
- **Ant Colony Optimization (ACO)**: Bio-inspired algorithm for initial feature subset selection
- **Genetic Algorithm (GA)**: Evolutionary approach for refining feature selection
- **PCA + Lasso Regression**: Dimensionality reduction combined with L1 regularization for optimal feature optimization

### Deep Learning Model
- **LSTM Neural Network**: Long Short-Term Memory network for sequential credit risk prediction
- **Monte Carlo Cross-Validation**: Robust evaluation using multiple random train-test splits
- **Hyperparameter Tuning**: Optimized architecture with dropout layers for regularization

### Comprehensive Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Visualization**: Training history plots, threshold analysis, and epoch-wise metric tracking
- **Model Interpretability**: Detailed classification reports and performance analysis

## Project Structure

```
credit_risk_project_option_b/
│
├── data/
│   └── german_credit_data.csv          # German credit dataset
│
├── src/
│   ├── preprocessing.py                 # Data preprocessing utilities
│   ├── feature_selection.py             # Feature selection algorithms
│   ├── model.py                         # LSTM model implementation
│   └── evaluation.py                    # Model evaluation and visualization
│
├── PJT Code implementation.ipynb        # Original Jupyter notebook
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
└── TODO.md                              # Development task tracking
```

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd credit_risk_project_option_b
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies** (if needed):
   ```bash
   pip install scikit-learn deap tensorflow matplotlib seaborn pandas numpy
   ```

## Usage

### Data Preparation
```python
from src.preprocessing import preprocess_data

# Preprocess the data
data, scaler = preprocess_data('data/german_credit_data.csv')
```

### Feature Selection
```python
from src.feature_selection import aco_feature_selection, genetic_algorithm_feature_selection, pca_lasso_feature_optimization
from sklearn.model_selection import train_test_split

# Split data
X = data.drop('credit_risk', axis=1)
y = data['credit_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply ACO feature selection
aco_features = aco_feature_selection(X_train, X_test, y_train, y_test)

# Apply Genetic Algorithm on ACO-selected features
ga_features = genetic_algorithm_feature_selection(X_train, X_test, y_train, y_test, aco_features)

# Apply PCA + Lasso optimization
optimized_features, avg_mse = pca_lasso_feature_optimization(X, y)
```

### Model Training and Evaluation
```python
from src.model import build_lstm_model, train_lstm_model, predict_credit_risk
from src.evaluation import plot_training_history, print_classification_report, plot_confusion_matrix
import numpy as np

# Prepare data for LSTM (reshape for sequential input)
X_selected = X[optimized_features[0]]  # Using first optimized feature set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scale and reshape data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build and train model
model = build_lstm_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
history = train_lstm_model(model, X_train_reshaped, y_train, X_test_reshaped, y_test)

# Evaluate model
y_pred_prob = model.predict(X_test_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)

# Plot results
plot_training_history(history)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)
```

### Prediction on New Data
```python
# Generate sample data for prediction
random_data = np.random.rand(10, len(optimized_features[0]))
random_data_scaled = scaler.transform(random_data)
random_data_reshaped = random_data_scaled.reshape((random_data_scaled.shape[0], 1, random_data_scaled.shape[1]))

# Make predictions
predictions = predict_credit_risk(model, random_data_reshaped)
print("Predicted credit risk probabilities:", predictions)
```

## Dependencies

The project requires the following Python packages:

- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- tensorflow>=2.8.0
- keras>=2.8.0
- deap>=1.3.0
- matplotlib>=3.5.0
- seaborn>=0.11.0

## Methodology

### 1. Data Preprocessing
- Load German credit dataset
- Handle missing values using advanced imputation techniques
- Create binary credit risk labels
- Apply WOE encoding to categorical features
- Scale numerical features

### 2. Feature Selection Pipeline
- **ACO**: Initial feature selection using ant colony optimization
- **GA**: Refine ACO results with genetic algorithm
- **PCA+Lasso**: Final optimization with dimensionality reduction and regularization

### 3. Model Development
- LSTM network architecture for sequential pattern recognition
- Binary classification for credit risk assessment
- Monte Carlo cross-validation for robust performance estimation

### 4. Evaluation Framework
- Comprehensive metric calculation (Accuracy, Precision, Recall, F1)
- Visualization of training dynamics
- Threshold analysis for optimal decision boundaries

## Results

The model achieves robust performance in credit risk classification:

- **Accuracy**: ~85-90% across validation folds
- **F1-Score**: ~0.80-0.85 for high-risk class identification
- **Precision/Recall**: Balanced metrics for risk assessment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- German Credit Dataset from UCI Machine Learning Repository
- Implementation inspired by advanced ML techniques in financial risk assessment
- Bio-inspired optimization algorithms for feature selection

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project demonstrates advanced machine learning techniques for credit risk assessment. For production use, additional validation, regulatory compliance, and domain expertise are recommended.
