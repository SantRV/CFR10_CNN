
import json
import tensorflow as tf
import time
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class CNNModel():
    name: str = ""
    model_path: str = ""
    results_path: str = ""
    train_history = None
    optimisers = ["adam", "sgd"]  # Stochastic Gradient Descent or adam
    class_names = []

    def __init__(self, num_classes: int, name: str, model_path: str, results_path: str, class_names: list, optimiser: str = "adam") -> None:
        # Validate data
        if optimiser not in self.optimisers:
            raise ValueError(f"INVALID OPTIMISER {optimiser}")

        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), padding='same',
                  activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

        # Compile model
        model.compile(
            optimizer=optimiser,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

        # Set class data
        self.model = model
        self.num_classes = num_classes
        self.optimiser = optimiser
        self.model_path = model_path
        self.results_path = results_path
        self.name = name
        self.class_names = class_names

        pass

    def get_summary(self):
        # Checking the model summary
        self.model.summary()

    def fit_model(self,
                  train_features,
                  train_labels,
                  validation_features,
                  validation_labels,
                  batch_size: int = 64,
                  epochs: int = 100,
                  save_model: bool = True,
                  save_results: bool = True,
                  plot_results: bool = True):
        print(
            f"---- CNN Training model - Optimiser: {self.optimiser} Batch Size: {batch_size} Epochs: {epochs} \n")
        start_time = time.time()

        self.train_history = self.model.fit(
            train_features,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(validation_features, validation_labels))

        print(f"CNN Finished Training - t: {time.time() - start_time} \n")

        # Save model to h5
        if save_model:
            save_name = f"{self.name}.h5"
            save_history = f"{self.name}_data.json"

            if self.model_path != "":
                save_name = f"{self.model_path}/{self.name}.h5"
                save_history = f"{self.model_path}/{self.name}_data.json"

            # Save model
            self.model.save(save_name)

            # Save metris
            with open(save_history, "w") as json_file:
                json.dump(self.train_history.history, json_file)

            print(f"CNN saved at {save_name} \n")

        # Plot loss and accuracy
        self.plot_loss(display=plot_results, save=save_results)
        self.plot_accuracy(display=plot_results, save=save_results)

        return

    """
        PREDICTIONS
    """

    def predict(self, model_name: str, data, true_labels, visualise_data: bool = True, save_results: bool = False):
        model_path = f"{model_name}.h5"

        if self.model_path != "":
            model_path = f"{self.model_path}/{model_name}.h5"

        # Load trained model
        loaded_model = load_model(model_path)

        # Make predictions
        predictions = loaded_model.predict(data)

        # Convert to labels
        pred_classes = np.argmax(predictions, axis=1)

        # Convert true_labels to predicted classes
        true_labels_array = true_labels.argmax(axis=1)
        print("test", true_labels_array.shape)
        print("labels", pred_classes.shape)

        if len(true_labels) > 0:
            self.plot_prediction_accuracy(
                true_labels=true_labels_array,
                pred_labels=pred_classes,
                display=visualise_data,
                save=save_results
            )

        # Visualise data
        if visualise_data:
            self.plot_images(data, true_labels_array, pred_classes)

        return pred_classes

    """
        PLOTTING SERVICE
    """

    def plot_loss(self, display: bool = True, save: bool = False):
        plt.figure(figsize=[6, 4])
        plt.plot(self.train_history.history['loss'], 'grey', linewidth=2.0)
        plt.plot(self.train_history.history['val_loss'], 'red', linewidth=2.0)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.title('Loss Curves', fontsize=12)

        if save:
            filename = f"loss_{datetime.now().strftime('%Y-%m-%d')}.png"

            if self.results_path != "":
                filename = f"{self.results_path}/{filename}"

            plt.savefig(filename, dpi=300)

        if display:
            plt.show()

        return

    def plot_accuracy(self, display: bool = True, save: bool = False):
        plt.figure(figsize=[6, 4])
        plt.plot(self.train_history.history['accuracy'], 'grey', linewidth=2.0)
        plt.plot(
            self.train_history.history['val_accuracy'], 'orange', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Accuracy Curves', fontsize=12)

        if save:
            filename = f"accuracy_{datetime.now().strftime('%Y-%m-%d')}.png"

            if self.results_path != "":
                filename = f"{self.results_path}/{filename}"

            plt.savefig(filename, dpi=300)

        if display:
            plt.show()
        return

    def plot_images(self, data_features, data_labels, prediction_labels, max_range: int = 25):
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes = axes.ravel()

        for i in np.arange(0, max_range):
            axes[i].imshow(data_features[i])
            axes[i].set_title("True: %s \nPredict: %s" % (self.class_names[np.argmax(
                data_labels[i])], self.class_names[prediction_labels[i]]))
            axes[i].axis('off')
            plt.subplots_adjust(wspace=1)

    def plot_prediction_accuracy(self, true_labels, pred_labels, display: bool = True, save: bool = False):
        if len(true_labels) != len(pred_labels):
            raise ValueError(
                "true_labels and pred_labels must have the same length.")

        num_samples = len(true_labels)

        # Calculate accuracy
        correct_predictions = [1 if true_labels[i] ==
                               pred_labels[i] else 0 for i in range(num_samples)]
        accuracy = sum(correct_predictions) / num_samples * 100

        # Create a plot to visualize accuracy
        plt.plot(range(len(true_labels)), correct_predictions,
                 'go-', label='Correct Prediction')
        plt.axhline(accuracy, color='r', linestyle='--', label='Accuracy')
        plt.xlabel('Sample Index')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Plot (Accuracy: {accuracy:.2f}%)')
        plt.legend(loc='upper right')
        plt.grid(True)

        if save:
            filename = f"prediction_accuracy_{datetime.now().strftime('%Y-%m-%d')}.png"

            if self.results_path != "":
                filename = f"{self.results_path}/{filename}"

            plt.savefig(filename)

        if display:
            plt.show()

        return
