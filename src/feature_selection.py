# feature selection utilities
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from deap import base, creator, tools, algorithms

def evaluate_features(selected_features, X_train, X_test, y_train, y_test):
    """Evaluate the performance of selected features using RandomForestRegressor."""
    if not selected_features:
        return 100000  # A high error if no features are selected

    # Train a random forest regressor on the selected features
    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf.fit(X_train.iloc[:, selected_features], y_train)

    # Make predictions and calculate mean squared error
    y_pred = clf.predict(X_test.iloc[:, selected_features])
    mse = mean_squared_error(y_test, y_pred)
    return mse

def aco_feature_selection(X_train, X_test, y_train, y_test, n_ants=10, n_iterations=10):
    """Perform Ant Colony Optimization for feature selection."""
    n_features = X_train.shape[1]
    best_feature_subset = []
    best_mse = float('inf')

    for _ in range(n_iterations):
        pheromone = np.ones(n_features)
        selected_features = []

        for _ in range(n_ants):
            available_features = [i for i in range(n_features) if i not in selected_features]

            # Check if there are available features to select
            if not available_features:
                break

            probabilities = pheromone[available_features] / sum(pheromone[available_features])
            selected_feature = np.random.choice(available_features, p=probabilities)
            selected_features.append(selected_feature)

        # Check if any features were selected by any ant
        if not selected_features:
            continue

        mse = evaluate_features(selected_features, X_train, X_test, y_train, y_test)

        if mse < best_mse:
            best_mse = mse
            best_feature_subset = selected_features

        # Update pheromone levels based on the best subset found
        pheromone[best_feature_subset] += 1 / best_mse

    return best_feature_subset

def genetic_algorithm_feature_selection(X_train, X_test, y_train, y_test, best_feature_subset, pop_size=50, cxpb=0.7, mutpb=0.2, ngen=20):
    """Perform Genetic Algorithm for feature selection on the ACO-selected features."""
    # Create a fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # Create the individual class
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize the genetic algorithm toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(best_feature_subset))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_features, X_train=X_train.iloc[:, best_feature_subset], X_test=X_test.iloc[:, best_feature_subset], y_train=y_train, y_test=y_test)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the genetic algorithm
    population = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=None, halloffame=None, verbose=True)

    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    selected_features = [best_feature_subset[i] for i, selected in enumerate(best_individual) if selected]

    return selected_features

def pca_lasso_feature_optimization(X, y, n_splits=5):
    """Perform PCA and Lasso for feature optimization using Monte Carlo Cross-Validation."""
    # Initialize PCA and Lasso models
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    lasso = LassoCV(cv=5)

    # Initialize lists to store selected features and MSE scores
    selected_features_optimized_list = []
    mse_scores = []

    # Perform Monte Carlo Cross-Validation
    for _ in range(n_splits):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(1000))

        # Standardize the selected features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform PCA for dimensionality reduction
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Perform Lasso (L1 regularization) for feature selection
        lasso.fit(X_train_pca, y_train)

        # Get the selected features after Lasso regularization
        selected_features_optimized = [col for col, coef in zip(X.columns, lasso.coef_) if coef != 0]

        # Evaluate the performance on the test set
        y_pred = lasso.predict(X_test_pca)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

        # Append the selected features to the list
        selected_features_optimized_list.append(selected_features_optimized)

    return selected_features_optimized_list, np.mean(mse_scores)
