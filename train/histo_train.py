import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import requests
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download and extract dataset (ensure the dataset URL is valid)
url = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/lung-and-colon-cancer-histopathological-images"
dataset_path = "lung_cancer_dataset"
zip_path = "dataset.zip"

if not os.path.exists(dataset_path):
    r = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    os.remove(zip_path)

# Set image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define data generators
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% for validation
)

train_generator = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Keep order for evaluation
)

# Select model
base_models = {
    'MobileNetV2': MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
}

best_model = None
best_acc = 0

for model_name, base_model in base_models.items():
    base_model.trainable = False  # Freeze pretrained layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator, epochs=10, validation_data=val_generator)
    val_acc = history.history['val_accuracy'][-1]

    model.save(f"{model_name.lower()}_model.keras")  # Save each model separately
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model

# Save the best model in the .keras format
best_model.save("best_model.keras")

# Evaluate on validation data
y_true = []
y_pred = []
class_labels = list(train_generator.class_indices.keys())

for images, labels in val_generator:
    preds = best_model.predict(images)
    y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to class index
    y_pred.extend(np.argmax(preds, axis=1))
    if len(y_true) >= val_generator.samples:
        break

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Per-class accuracy
class_accuracies = {}
y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)

for i, label in enumerate(class_labels):
    correct_preds = np.sum((y_true_np == i) & (y_pred_np == i))
    total_preds = np.sum(y_true_np == i)
    class_accuracies[label] = correct_preds / total_preds if total_preds > 0 else 0

print("\nPer-Class Accuracy:")
for label, acc in class_accuracies.items():
    print(f"{label}: {acc:.2%}")

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Model Accuracy: {accuracy:.2%}")
