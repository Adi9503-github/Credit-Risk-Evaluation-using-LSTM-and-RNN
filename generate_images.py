import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Sample data for demonstration
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Sample Confusion Matrix')
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
plt.savefig('images/confusion_matrix.png')
plt.close()

# Generate Training History Plot
epochs = range(1, 11)
train_acc = [0.5 + 0.4 * (1 - np.exp(-i/5)) for i in epochs]
val_acc = [0.45 + 0.35 * (1 - np.exp(-i/6)) for i in epochs]
train_loss = [1.0 - 0.08 * i for i in epochs]
val_loss = [1.1 - 0.07 * i for i in epochs]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('images/training_history.png')
plt.close()

print("Sample images generated successfully!")
