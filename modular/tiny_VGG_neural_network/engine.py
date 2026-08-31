import torch
from torch import nn
from utils import device

def train_step(
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module =  nn.CrossEntropyLoss()
        ):
    """
    Performs one complete training pass over the training dataset.

    Args:
        model: The PyTorch model being trained.
        train_dataloader: DataLoader containing the training batches.
        optimizer: Optimizer used to update the model parameters.
        loss_fn: Loss function used to calculate prediction error.

    Returns:
        None
    """ 
    model.train()
    
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch: {batch} | Train Loss: {loss}")        
    
    
def test_step(
        model: torch.nn.Module, 
        test_dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module =  nn.CrossEntropyLoss()):
    """
    Evaluates the model on the test dataset without updating its parameters.

    Args:
        model: The PyTorch model being evaluated.
        test_dataloader: DataLoader containing the test batches.
        loss_fn: Loss function used to calculate prediction error.

    Returns:
        None
    """
    
    model.eval()
 
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            print(f"Batch: {batch} | Test Loss: {loss}")        
    

def train(epochs:int, model, train_dataloader, test_dataloader, optimizer, loss_fn):
    """
    Trains the model for a specified number of epochs and evaluates it
    on the test dataset after each epoch.

    Args:
        epochs: Number of times the model will iterate over the training dataset.
        model: The PyTorch model to train.
        train_dataloader: DataLoader containing the training data.
        test_dataloader: DataLoader containing the test data.
        optimizer: Optimizer used to update model parameters.

    Returns:
        None
    """
    for epoch in range(epochs):
        print(f"Epoch: {epoch +1}")
        train_step(model=model, train_dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_step(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn)


# train(epochs=100,model=modelV0, train_dataloader=train_dataloader, optimizer=optimizer )
