# evaluation utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_metrics_vs_threshold(y_true, y_pred_prob, thresholds=np.linspace(0, 1, 100)):
    """Plot precision, recall, and F1 score vs threshold."""
    precision_values = [precision_score(y_true, (y_pred_prob > threshold).astype(int)) for threshold in thresholds]
    recall_values = [recall_score(y_true, (y_pred_prob > threshold).astype(int)) for threshold in thresholds]
    f1_values = [f1_score(y_true, (y_pred_prob > threshold).astype(int)) for threshold in thresholds]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(thresholds, precision_values, label='Precision')
    plt.title('Precision vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(thresholds, recall_values, label='Recall')
    plt.title('Recall vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(thresholds, f1_values, label='F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_epoch_wise_metrics(model, X_train, y_train, X_val, y_val, epochs=10):
    """Plot precision, recall, and F1 score over epochs."""
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []

    for epoch in range(epochs):
        # Train for one epoch
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

        # Calculate metrics on training set
        y_train_pred = (model.predict(X_train) > 0.5).astype(int)
        train_precision.append(precision_score(y_train, y_train_pred))
        train_recall.append(recall_score(y_train, y_train_pred))
        train_f1.append(f1_score(y_train, y_train_pred))

        # Calculate metrics on validation set
        y_val_pred = (model.predict(X_val) > 0.5).astype(int)
        val_precision.append(precision_score(y_val, y_val_pred))
        val_recall.append(recall_score(y_val, y_val_pred))
        val_f1.append(f1_score(y_val, y_val_pred))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs+1), train_precision, label='Training Precision', color='blue')
    plt.plot(range(1, epochs+1), val_precision, label='Validation Precision', color='orange')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs+1), train_recall, label='Training Recall', color='blue')
    plt.plot(range(1, epochs+1), val_recall, label='Validation Recall', color='orange')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs+1), train_f1, label='Training F1 Score', color='blue')
    plt.plot(range(1, epochs+1), val_f1, label='Validation F1 Score', color='orange')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Low Risk', 'High Risk'])
    plt.yticks(tick_marks, ['Low Risk', 'High Risk'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()
