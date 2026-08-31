from torch.utils.data import DataLoader
from torchvision import datasets, transforms # transorm is for converting the img data to pytorch tensors.

def transform_dataset(train_dataset_directory, test_dataset_directory):
    """
        Input data would be direct blob images.
        train_dataset_directory: directory path for training data
        test_dataset_directory: directory path for test data
        
        output:
        
        In return you will get a transformed version of your images in Tensor, either flip (50% probability) and would scale down to size 64x64. 
        and also the list of classes names present in the dataset.
    """
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)), # reducing img pexels to a fix size.
        transforms.RandomHorizontalFlip(p=0.5), # randomly flipping image on probability of 50%
        transforms.ToTensor() # to convert data into "Tensor"
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dataset_directory,
        transform=data_transform,
        target_transform=None
    )

    test_dataset = datasets.ImageFolder(
        root=test_dataset_directory,
        transform=data_transform,
        target_transform=None
    )
    
    return train_dataset, test_dataset, train_dataset.classes

    
    


def create_dataloaders(train_dataset, test_dataset, batch_size, num_workers, shuffle = True ):
    """
        This require configuration for dataloader. In what way you want your data to be split and grouped together for your model training, and shuffling data helps to train the model well. So use it wisely according to your machine power. Num_workers in simple terms is nothing but how many CPU threads do you want to use to load your data into RAM.
    """

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return train_dataloader, test_dataloader
