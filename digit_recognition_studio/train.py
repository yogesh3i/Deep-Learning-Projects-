import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ensure output assets directory exists
os.makedirs('static/assets', exist_ok=True)

print("="*60)
print("1. LOADING AND PREPROCESSING MNIST DATASET...")
print("="*60)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] and reshape to (28, 28, 1) for Conv2D
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape:  {X_test.shape}")

print("="*60)
print("2. BUILDING CONVOLUTIONAL NEURAL NETWORK (CNN)...")
print("="*60)

# Build custom robust CNN model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_layer_1'),
    layers.MaxPooling2D((2, 2), name='maxpool_layer_1'),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer_2'),
    layers.MaxPooling2D((2, 2), name='maxpool_layer_2'),
    
    # Dropout to prevent overfitting
    layers.Dropout(0.25, name='dropout_1'),
    
    # Flattening & Fully Connected Layers
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='dense_1'),
    layers.Dropout(0.5, name='dropout_2'),
    layers.Dense(10, activation='softmax', name='output_layer')
])

model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("="*60)
print("3. TRAINING MODEL...")
print("="*60)

# Train the model (5 epochs is extremely fast and reaches ~98.5%+ accuracy)
history = model.fit(
    X_train, y_train,
    epochs=6,
    batch_size=128,
    validation_data=(X_test, y_test)
)

print("="*60)
print("4. EVALUATING MODEL...")
print("="*60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")

# Save the trained model
model_path = 'digit_model.h5'
model.save(model_path)
print(f"Saved trained model to {model_path}")

print("="*60)
print("5. GENERATING EVALUATION ASSETS...")
print("="*60)

# 5.1 Training History Curves (Loss & Accuracy)
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#06b6d4', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#8b5cf6', linewidth=2)
ax1.set_title('Model Accuracy Progress', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.15)

# Plot loss
ax2.plot(history.history['loss'], label='Train Loss', color='#ef4444', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', linewidth=2)
ax2.set_title('Model Loss Progress', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.15)

plt.tight_layout()
curves_path = 'static/assets/training_curves.png'
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved training history curves to {curves_path}")

# 5.2 Confusion Matrix Heatmap
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Purples', 
    xticklabels=list(range(10)), 
    yticklabels=list(range(10)),
    cbar=False
)
plt.title('Test Dataset Confusion Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Digit Label', fontsize=12, labelpad=10)
plt.ylabel('Actual Digit Label', fontsize=12, labelpad=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

cm_path = 'static/assets/confusion_matrix.png'
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved confusion matrix heatmap to {cm_path}")

# Print text-based classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("="*60)
print("TRAINING PROCESS COMPLETED SUCCESSFULLY!")
print("="*60)
