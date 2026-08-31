import torch
import argparse
from engine import train
from data_setup import transform_dataset, create_dataloaders
from model_builder import TinyVGGConvolutionalModel
from utils import device


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a TinyVGG model on the pizza, steak, and sushi dataset."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images in each batch."
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=10,
        help="Number of hidden units in the model."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer."
    )

    return parser.parse_args()


args = get_args()


# Hyperparameters
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate

# Setup directories
train_dir = "../../datasets/pizza_steak_sushi/train"
test_dir = "../../datasets/pizza_steak_sushi/test"




train_dataset, test_dataset , class_names = transform_dataset(
    train_dataset_directory=train_dir,
    test_dataset_directory=test_dir
)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader = create_dataloaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=1
)

# Create model with help from model_builder.py
model = TinyVGGConvolutionalModel(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
train(model=model,
             epochs=NUM_EPOCHS,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn
             )

