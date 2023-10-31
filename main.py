

from cnn.cnn_model import CNNModel
from cnn.cnn_model_v2 import CNNModelV2
from cnn.cnn_model_v3 import CNNModelV3
from data_loader.data_loader_service import DataLoaderService
import numpy as np


def main():
    # Load data
    data_path = "./data"
    model_path = "./models"
    results_path = "./results"
    epochs = 120
    batch_size = 64
    model_name = f"v1_epochs_{epochs}_batch_{batch_size}"
    model_name_V2 = f"v2_epochs_{epochs}_batch_{batch_size}"
    model_name_V3 = f"v3_epochs_{epochs}_batch_{batch_size}"

    print(f" ----- Starting CNN {model_name_V2} for CIFAR10 ----- \n")
    dataService = DataLoaderService(data_path)

    # Get data
    train_images, train_labels, valid_images, valid_labels = dataService.load_data()
    test_images, test_labels = dataService.get_testing_set()

    # Get loaded image classes
    class_name = dataService.get_classes()

    # Build CNN model
    # cnnModel = CNNModel(name=model_name, model_path=model_path,
    #                     results_path=results_path, num_classes=len(class_name),
    #                     class_names=class_name)

    cnnModel = CNNModelV2(name=model_name_V2, model_path=model_path,
                          results_path=results_path, num_classes=len(
                              class_name),
                          class_names=class_name, optimiser="sgd")

    # cnnModel = CNNModelV3(name=model_name_V3, model_path=model_path,
    #                       results_path=results_path, num_classes=len(
    #                           class_name),
    #                       class_names=class_name, optimiser="adam")

    cnnModel.get_summary()

    # Train model
    cnnModel.fit_model(
        train_features=train_images,
        train_labels=train_labels,
        validation_features=valid_images,
        validation_labels=valid_labels,
        batch_size=batch_size,
        epochs=epochs,
        save_model=True,
        save_results=True,
        plot_results=True)

    # Predict
    cnnModel.predict(
        model_name=model_name_V2,
        data=test_images,
        true_labels=test_labels,
        visualise_data=True,
        save_results=True
    )

    print(f"End CNN {model_name} \n")

    exit()


main()
