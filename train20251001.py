import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix as cm,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, regularizers as R
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model

from ultralytics import YOLO


def image_to_numpy(image_path: Path):
    image = cv2.imread(str(image_path))
    return np.array(image)


def load_images(directory: Path):
    images = sorted(list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg")))
    if len(images) >= 2:
        front_image = image_to_numpy(images[0])
        side_image = image_to_numpy(images[1])
        return (front_image, side_image)
    else:
        return None


def pad_and_clip(x1, y1, x2, y2, width, height, pad_ratio=1):
    padding_width = int((x2 - x1) * pad_ratio)
    padding_height = int((y2 - y1) * 1)

    x1 = max(0, x1 - padding_width)
    x2 = min(width, x2 + padding_width)
    y1 = max(0, y1 - padding_height)
    y2 = min(height, y2 + padding_height)

    return x1, y1, x2, y2


def crop_with_yolo(image: np.ndarray, sensitivity=0.1, pad_ratio=0.1):
    """Detect vertebrae with YOLO and mask the image to keep only detected regions."""
    height, width = image.shape[:2]
    result = yolo_model.predict(source=image, conf=sensitivity, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No box detected")
        return image

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = pad_and_clip(x1, y1, x2, y2, width, height, pad_ratio)
        mask[y1:y2, x1:x2] = 255

    out = np.zeros_like(image)
    out[mask == 255] = image[mask == 255]
    return out


def normalize_image(image: np.ndarray, config):
    """Grayscale, resize, and apply ResNet50 preprocessing."""
    if config.grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.stack([image, image, image], axis=-1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (config.target_size[1], config.target_size[0]))
    image = image.astype("float32")
    image = resnet50.preprocess_input(image)
    return image


def prepare_images(directory: Path, label, config):
    """Load, crop, and normalize image pairs from subdirectories."""
    pairs = []
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    load_amount = max(int(len(subdirs) * config.proportion), 1)
    subdirs = subdirs[:load_amount]

    for subdir in subdirs:
        images = load_images(subdir)
        if images is None:
            continue
        else:
            front_image, side_image = load_images(subdir)
        if front_image is None or side_image is None:
            continue
        front_image = crop_with_yolo(front_image)
        side_image = crop_with_yolo(side_image)

        front_image = normalize_image(front_image, config=config)
        side_image = normalize_image(side_image, config=config)
        pairs.append((front_image, side_image, label, subdir.name))
    return pairs


def load_image_pairs(config):
    """Load fracture (label=1) and non-fracture (label=0) image pairs."""
    pairs = []
    pairs += prepare_images(config.fracture_dir, 1, config)
    pairs += prepare_images(config.non_fracture_dir, 0, config)
    return pairs


@dataclass
class Config:
    target_size: tuple = (400, 300)
    grayscale: bool = True
    depth: int = 2
    nodes: int = 8
    l2: float = 1e-4
    sigmoid_temperature: float = 2.0
    dropout: float = 0.3
    proportion: float = 1.0
    fracture_dir: Path = Path("./data/fracture")
    non_fracture_dir: Path = Path("./data/non_fracture")
    yolo_model_path: Path = Path("./best.pt")
    enable_save_param: bool = True
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_path = Path(f"./log/{self.timestamp}/")
        self.log_path.mkdir(exist_ok=True)


def create_model(config):
    """Build a two-input ResNet50-based model for fracture detection."""
    base_model = ResNet50(weights="imagenet", include_top=False, pooling=None)
    base_model.trainable = False
    l2, depth, nodes, dropout = config.l2, config.depth, config.nodes, config.dropout

    def neck(x):
        x = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, kernel_regularizer=R.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_regularizer=R.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        gap = layers.GlobalAveragePooling2D()(x)
        gmp = layers.GlobalMaxPooling2D()(x)
        return layers.Concatenate()([gap, gmp])

    # Front image branch
    input_front = Input(shape=(config.target_size[0], config.target_size[1], 3), name="front_input")
    x_front = base_model(input_front)
    x_front = neck(x_front)

    # Side image branch
    input_side = Input(shape=(config.target_size[0], config.target_size[1], 3), name="side_input")
    x_side = base_model(input_side)
    x_side = neck(x_side)

    # Merge and classify
    x = Concatenate()([x_front, x_side])

    for _ in range(depth):
        x = Dense(nodes, activation="relu", kernel_regularizer=R.l2(l2))(x)
        x = Dropout(dropout)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer=R.l2(l2))(x)

    model = Model(inputs=[input_front, input_side], outputs=output)
    return model


if __name__ == "__main__":
    config = Config(l2=1e-3)
    print("timestamp: ", config.timestamp)
    yolo_model = YOLO(config.yolo_model_path)
    timestamp = config.timestamp

    # Load and split data
    all_pairs = load_image_pairs(config)
    X_front = np.array([pair[0] for pair in all_pairs])
    X_side = np.array([pair[1] for pair in all_pairs])
    y = np.array([pair[2] for pair in all_pairs])
    directory_names = np.array([pair[3] for pair in all_pairs])

    X_front_train, X_front_test, X_side_train, X_side_test, directory_name_train, directory_name_test, y_train, y_test \
        = train_test_split(X_front, X_side, directory_names, y, test_size=0.2, random_state=27)

    print("Front: ", X_front_train.shape)
    print("Side: ", X_side_train.shape)
    print("Output: ", y_train.shape)
    print("y mean: ", y_train.mean())

    # Save test set directory names
    with open(config.log_path / f"fracture_directory_names{timestamp}.txt", "w") as f, \
         open(config.log_path / f"non_fracture_directory_names{timestamp}.txt", "w") as n:
        for y_val, d in zip(y_test, directory_name_test):
            if y_val == 1:
                f.write(d + "\n")
            else:
                n.write(d + "\n")

    # Build and compile model
    model = create_model(config)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC", "Precision", "Recall"])
    model.summary()

    # Train
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print("class_weight: ", class_weight)

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        [X_front_train, X_side_train], y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[es],
        verbose=1,
    )

    # Save model
    if config.enable_save_param:
        model.save(config.log_path / f"detect_fracture_{timestamp}.keras")
        with open(config.log_path / f"config_{timestamp}.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

    model.save_weights(f"weight{timestamp}.weights.h5")

    # Evaluate
    score = model.evaluate([X_front_test, X_side_test], y_test)
    print(f"Test Loss: {score[0]}")
    print(f"Test Accuracy: {score[1]}")

    y_pred_proba = model.predict([X_front_test, X_side_test])
    y_pred = (y_pred_proba > 0.5).astype("int32").flatten()

    print(cm(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Training history plots
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # ROC curve and optimal threshold
    y_true = y_test.astype(int)
    pos_score = np.ravel(y_pred_proba)

    fpr, tpr, thresholds = roc_curve(y_true, pos_score)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    youdenJ = tpr - fpr
    best_idx = np.argmax(youdenJ)
    best_thr = thresholds[best_idx]
    print("Best threshold by Youden's J:", best_thr)

    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC curve")
    plt.show()

    y_pred_best = (pos_score > best_thr).astype(int)
    print(cm(y_true, y_pred_best))
    print(classification_report(y_true, y_pred_best, digits=3))
