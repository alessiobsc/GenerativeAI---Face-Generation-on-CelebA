from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import os

class CelebAConditional(datasets.CelebA):
    """Wrapper to extract only the required attributes: Male(20), Smiling(31), Young(39)."""
    def __getitem__(self, index):
        img, all_attrs = super().__getitem__(index)
        # Extract specific attributes and convert to float for the network
        specific_attrs = all_attrs[[20, 31, 39]].float()
        return img, specific_attrs

def get_dataloaders(data_dir, batch_size=128, num_workers=4):
    """Prepares and returns training and testing dataloaders."""
    logging.info(f"Loading CelebA dataset from: {data_dir}")
    if not os.path.exists(data_dir):
        logging.warning(f"Data directory {data_dir} does not exist. Dataset initialization might fail.")
    
    # Transforms: resize to 64x64 and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Note: On HPC, download should be False as data is already accessible locally.
    logging.info("Initializing train dataset...")
    train_dataset = CelebAConditional(root=data_dir, split='train', target_type='attr',
                                      transform=transform, download=False)
    logging.info("Initializing test dataset...")
    test_dataset = CelebAConditional(root=data_dir, split='test', target_type='attr',
                                     transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader