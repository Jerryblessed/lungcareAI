#!/usr/bin/env python3
import os, zipfile, numpy as np, tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1) DOWNLOAD & UNZIP
ZIP_URL    = "https://www.kaggle.com/api/v1/datasets/download/mohamedhanyyy/chest-ctscan-images"
ZIP_PATH   = "chest_ctscan.zip"
TARGET_DIR = "ct_data"

os.system(f'curl -L -o {ZIP_PATH} "{ZIP_URL}"')
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(TARGET_DIR)
os.remove(ZIP_PATH)

# 2) AUTO-DETECT split root
def find_split_root(base_path):
    for entry in os.listdir(base_path):
        full = os.path.join(base_path, entry)
        if os.path.isdir(full):
            if {'train','valid','test'}.issubset(set(os.listdir(full))):
                return full
    return base_path

SPLIT_ROOT = find_split_root(TARGET_DIR)
print(f"> Using split root: {SPLIT_ROOT}")

TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")
VALID_DIR = os.path.join(SPLIT_ROOT, "valid")
TEST_DIR  = os.path.join(SPLIT_ROOT, "test")

# ---------------------------
# 2) HYPERPARAMS & DATALOADERS
# ---------------------------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 10

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

# ---------------------------
# 3) BUILD & COMPILE MODEL
# ---------------------------
print("> Building DenseNet121 model …")
base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ---------------------------
# 4) TRAIN
# ---------------------------
print(f"\n> Training for {EPOCHS} epochs …")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# Save the trained model
OUT_MODEL = "ctscan_densenet121.keras"
model.save(OUT_MODEL)
print(f"> Model saved to {OUT_MODEL}")

# ---------------------------
# 5) EVALUATE ON TEST SET
# ---------------------------
print("\n> Evaluating on test set …")
y_true = []
y_pred = []
class_labels = list(test_gen.class_indices.keys())

for i in range(len(test_gen)):
    x_batch, y_batch = test_gen[i]
    preds = model.predict(x_batch, verbose=0)
    y_true.append(np.argmax(y_batch))
    y_pred.append(np.argmax(preds))

# 5.1 Classification report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 5.2 Confusion matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))

# 5.3 Overall accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Test Accuracy: {acc:.2%}")

# 5.4 Per-class accuracy
print("\nPer-class Accuracy:")
y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
for idx, cls in enumerate(class_labels):
    total   = np.sum(y_true_np == idx)
    correct = np.sum((y_true_np == idx) & (y_pred_np == idx))
    pct     = (correct / total) if total > 0 else 0
    print(f"  {cls}: {pct:.2%}")

# ---------------------------
# 6) OPTIONAL FINE-TUNING
# ---------------------------
# To unfreeze and fine-tune, uncomment:
# base.trainable = True
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_gen, epochs=5, validation_data=val_gen)


# --- Confusion Matrix ---
# [[109   6   0   5]
#  [ 27  19   1   4]
#  [  1   0  53   0]
#  [ 40   1   0  49]]

# Overall Test Accuracy: 73.02%

# Per-class Accuracy:
#   adenocarcinoma: 90.83%
#   large.cell.carcinoma: 37.25%
#   normal: 98.15%
#   squamous.cell.carcinoma: 54.44%