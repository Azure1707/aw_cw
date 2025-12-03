import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. LOAD AND SPLIT DATA
def load_and_split_dataset(file_location):
    """Load images and split into train/validation/test sets.

    Parameters:
    file_location: Path to the data set.

    Returns:
    train_ds: Training data set (70% split).
    val_ds: Validation data set (15% split).
    test_ds: Test data set (15% split).
    class_names: List of class names.
    """

    # Load images from directory.
    # - "Resize" images to 64x64 as Keras defaults the input to 256x256?
    # - Batch size of 128 to allow for faster GPU processing.
    # - Shuffle with customisable seed to allow for reproducibility.
    dataset = keras.utils.image_dataset_from_directory(
        file_location,
        image_size=(64, 64),
        batch_size=128,
        shuffle=True,
        seed=123
    )

    class_names = dataset.class_names
    dataset_size = len(dataset)

    # Data set splits: 70% training, 15% validation, 15% testing.
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)

    train_ds = dataset.take(train_size)
    remaining_ds = dataset.skip(train_size)
    val_ds = remaining_ds.take(val_size)
    test_ds = remaining_ds.skip(val_size)

    # Performance optimisation recommendations from Apxml.com.
    # - cache(): Stores data set in GPU memory after the first epoch has
    # passed.
    # - prefetch(AUTOTUNE): Overlaps the preprocessing and model execution,
    # while the GPU trains on batch N, CPU prepares batch N+1.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(f"Classes: {class_names}")
    print(f"Split: {train_size} train, {val_size} val, {dataset_size - train_size - val_size} test batches")

    return train_ds, val_ds, test_ds, class_names

# 2. BUILD CNN MODEL
def build_cnn_model(use_augmentation=True):
    """Build the CNN model. Code was inspired by references in Lab 7,
    where we adapted code snippets to our project. "adam" was chosen as the
    optimiser as it provided us with greater accuracy.

    Parameters:
    use_augmentation: Whether to use data augmentation or not (default: True)

    Returns:
    model: Compiled Keras Sequential model.
    """

    layers = [
        keras.layers.Input(shape=(64, 64, 3)),

        # Normalise pixels from 0-255 to 0-1.
        keras.layers.Rescaling(1. / 255),
    ]

    if use_augmentation:
        layers.extend([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomContrast(0.2),
        ])

    layers.extend([
        # First convolutional block.
        keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding="same"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Second convolutional block.
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding="same"),
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1),
                            activation='relu', padding="same"),
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1),
                            activation='relu', padding="same"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Dropout to reduce overfitting.
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation='softmax')
    ])

    model = keras.models.Sequential(layers)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

# 3. TRAIN MODEL
def train_model(model, train_ds, val_ds):
    """Train the CNN model on the training data set.

    Parameters:
    model: Compiled Keras model from above.
    train_ds: Training data set.
    val_ds: Validation data set.

    Returns:
    history: Training history metrics for plotting.
    """

    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds
    )

    return history

# 4. PLOT TRAINING HISTORY
def plot_training_history(history):
    """Plot training and validation loss/accuracy curves.

    Parameters:
    history: Training history returned from model.fit().

    Returns:
    None (saves plot to 'outputs/training_curves.png')
    """

    f, ax = plt.subplots(2, 1, figsize=(5, 5))

    # Loss curves.
    ax[0].plot(history.history['loss'], color='b', label='Training Loss')
    ax[0].plot(history.history['val_loss'], color='r', label='Validation Loss')
    ax[0].legend(loc="upper right")

    # Accuracy curves.
    ax[1].plot(history.history['accuracy'], color='b', label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.show()

    print(f'Best Validation Accuracy: {np.max(history.history["val_accuracy"]):.4f}')

# 5. EVALUATE MODEL
def evaluate_model_confusion(model, test_ds, class_names):
    """Evaluate the model on the test data set and plot a confusion matrix.

    Parameters:
        model: Trained Keras model
        test_ds: Test data set
        class_names: List of class names.

    returns:
        None (Displays confusion matrix plot)
    """
    # Extract true labels and predictions.
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    # Generate confusion matrix.
    cm = confusion_matrix(y_true, y_pred)

    # Display confusion matrix.
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: CNN model')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()

    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    choice = input("Enable data augmentation? (y/n): ").lower()
    use_augmentation = choice != 'n'

    # 1. LOAD AND SPLIT DATA
    print("1. Loading dataset...")
    train_ds, val_ds, test_ds, class_names = load_and_split_dataset(
        "EuroSAT")

    # 2. BUILD CNN MODEL
    print(f"\n2. Building CNN model... (aug: {use_augmentation})")
    model = build_cnn_model(use_augmentation=use_augmentation)
    model.summary()

    # 3. TRAIN MODEL
    print("\n3. Training model...")
    history = train_model(model, train_ds, val_ds)

    # 4. PLOT TRAINING HISTORY
    print("\n4. Plotting training history...")
    plot_training_history(history)
    print("\nModel has been saved to 'outputs/training_curves.png'")

    # 5. EVALUATE MODEL
    print("\n5. Evaluating model...")
    evaluate_model_confusion(model, test_ds, class_names)
    print("\nModel has been saved to 'outputs/confusion_matrix.png'")