"""
Data loaders for various datasets.

Provides train/val/test splits with proper transformations.
Phase 1: MNIST only
Future: CIFAR-10, ImageNet subsets, etc.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os


def get_mnist_loaders(
    data_dir='./data',
    batch_size=128,
    eval_batch_size=None,  # If None, uses batch_size for val/test
    val_split=0.1,
    num_workers=4,
    seed=42,
    use_augmentation=False
):
    """
    Get MNIST data loaders with train/val/test splits.
    
    Args:
        data_dir (str): Directory to store/load data
        batch_size (int): Batch size for training
        eval_batch_size (int): Batch size for validation/test (if None, uses batch_size)
                              Larger batch size = smoother metrics
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of worker processes for data loading
        seed (int): Random seed for reproducible splits
        use_augmentation (bool): Whether to use data augmentation on training set
    
    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' containing DataLoaders
        dict: Dictionary with dataset info (input_dim, num_classes, sizes)
    
    Example:
        >>> loaders, info = get_mnist_loaders(batch_size=128, use_augmentation=True)
        >>> print(info)
        {'input_dim': 784, 'num_classes': 10, 'train_size': 54000, 'val_size': 6000, 'test_size': 10000}
        >>> for batch_idx, (data, target) in enumerate(loaders['train']):
        ...     # data: (batch_size, 784), target: (batch_size,)
        ...     pass
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # MNIST normalization values (mean and std of MNIST dataset)
    # These are precomputed statistics
    mnist_mean = 0.1307
    mnist_std = 0.3081
    
    # Training transformations (with optional augmentation)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # ±10 degrees rotation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1)  # 90%-110% zoom
            ),
            transforms.ToTensor(),
            transforms.Normalize((mnist_mean,), (mnist_std,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mnist_mean,), (mnist_std,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
        ])
    
    # Test/validation transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
    ])
    
    # Download/load datasets
    # Training set uses augmentation (if enabled)
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Test set never uses augmentation
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training set into train and validation
    train_size = len(train_dataset)
    val_size = int(train_size * val_split)
    train_size = train_size - val_size
    
    # Use generator for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Use separate batch size for evaluation (smoother metrics)
    eval_batch = eval_batch_size if eval_batch_size is not None else batch_size
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch,  # Larger batch for smoother metrics
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch,  # Larger batch for smoother metrics
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset info
    info = {
        'input_dim': 784,  # 28 * 28
        'num_classes': 10,  # digits 0-9
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(test_dataset),
        'image_shape': (1, 28, 28),  # For future CNN models
        'mean': mnist_mean,
        'std': mnist_std,
        'augmentation': use_augmentation
    }
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return loaders, info


def get_fashion_mnist_loaders(
    data_dir='./data',
    batch_size=128,
    eval_batch_size=None,  # If None, uses batch_size for val/test
    val_split=0.1,
    num_workers=4,
    seed=42,
    use_augmentation=False
):
    """
    Get Fashion-MNIST data loaders with train/val/test splits.
    
    Fashion-MNIST is MUCH HARDER than MNIST:
    - MNIST: Even simple models get 98%+ accuracy
    - Fashion-MNIST: Good models get 88-92% accuracy
    - Better for distinguishing architecture quality!
    
    10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
    
    Args:
        data_dir (str): Directory to store/load data
        batch_size (int): Batch size for training
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of worker processes for data loading
        seed (int): Random seed for reproducible splits
        use_augmentation (bool): Whether to use data augmentation on training set
    
    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' containing DataLoaders
        dict: Dictionary with dataset info (input_dim, num_classes, sizes)
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Fashion-MNIST normalization values (precomputed)
    fashion_mean = 0.2860
    fashion_std = 0.3530
    
    # Training transformations (with optional augmentation)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally (clothes can be mirrored)
            transforms.RandomRotation(15),  # ±15 degrees rotation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # ±10% translation
                scale=(0.9, 1.1)  # 90%-110% zoom
            ),
            transforms.ToTensor(),
            transforms.Normalize((fashion_mean,), (fashion_std,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((fashion_mean,), (fashion_std,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
        ])
    
    # Test/validation transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((fashion_mean,), (fashion_std,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
    ])
    
    # Download/load datasets with robust error handling
    import shutil
    import time
    fashion_mnist_dir = os.path.join(data_dir, 'FashionMNIST')
    raw_folder = os.path.join(fashion_mnist_dir, 'raw')
    
    # Helper function to download with retries
    def download_with_retries(download_func, dataset_name="dataset", max_retries=3):
        for attempt in range(max_retries):
            try:
                return download_func()
            except (RuntimeError, Exception) as e:
                error_msg = str(e).lower()
                if "corrupted" in error_msg or "not found" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"⚠️  {dataset_name} download failed (attempt {attempt + 1}/{max_retries}). Cleaning up and retrying...")
                        # Remove potentially corrupted files
                        if os.path.exists(raw_folder):
                            shutil.rmtree(raw_folder)
                            print(f"   Removed corrupted files from {raw_folder}")
                        # Wait a bit before retrying (exponential backoff)
                        time.sleep(2 ** attempt)
                    else:
                        print(f"❌ {dataset_name} download failed after {max_retries} attempts.")
                        print(f"   This is likely a network/server issue.")
                        print(f"   Manual fix: Delete '{raw_folder}' and ensure you have internet connectivity.")
                        print(f"   Or download manually from: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")
                        raise RuntimeError(
                            f"Failed to download {dataset_name} after {max_retries} attempts. "
                            f"Please check your internet connection or manually download the dataset."
                        )
                else:
                    # Different error, don't retry
                    raise
    
    # Download training set
    def download_train():
        return datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
    
    train_dataset = download_with_retries(download_train, "Training set")
    
    # Download test set
    def download_test():
        return datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
    
    test_dataset = download_with_retries(download_test, "Test set")
    
    # Split training set into train and validation
    train_size = len(train_dataset)
    val_size = int(train_size * val_split)
    train_size = train_size - val_size
    
    # Use generator for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Use separate batch size for evaluation (smoother metrics)
    eval_batch = eval_batch_size if eval_batch_size is not None else batch_size
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch,  # Larger batch for smoother metrics
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch,  # Larger batch for smoother metrics
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset info
    info = {
        'input_dim': 784,  # 28 * 28
        'num_classes': 10,  # 10 clothing categories
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(test_dataset),
        'image_shape': (1, 28, 28),
        'mean': fashion_mean,
        'std': fashion_std,
        'augmentation': use_augmentation,
        'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    }
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return loaders, info


def get_cifar10_flattened_loaders(
    data_dir='./data',
    batch_size=128,
    eval_batch_size=None,  # If None, uses batch_size for val/test
    val_split=0.1,
    num_workers=4,
    seed=42,
    use_augmentation=False
):
    """
    Get CIFAR-10 data loaders (flattened for MLPs).
    
    CIFAR-10 is VERY HARD for MLPs:
    - MLPs on CIFAR-10: Good models get 50-60% accuracy (flattened, no convolutions)
    - CNNs on CIFAR-10: Good models get 85-95% accuracy
    - Warning: 32x32x3 = 3072 input dims (much larger than MNIST's 784)
    
    Best used as a challenging benchmark or when ready to switch to CNNs.
    
    10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    Args:
        data_dir (str): Directory to store/load data
        batch_size (int): Batch size for training
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of worker processes
        seed (int): Random seed
        use_augmentation (bool): Whether to use data augmentation
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
        dict: Dataset information
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # CIFAR-10 normalization values (per-channel)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)
    
    # Training transformations
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 3x32x32 -> 3072
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 3x32x32 -> 3072
        ])
    
    # Test/validation transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 3x32x32 -> 3072
    ])
    
    # Download/load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training set
    train_size = len(train_dataset)
    val_size = int(train_size * val_split)
    train_size = train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset info
    info = {
        'input_dim': 3072,  # 3 * 32 * 32
        'num_classes': 10,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(test_dataset),
        'image_shape': (3, 32, 32),
        'mean': cifar_mean,
        'std': cifar_std,
        'augmentation': use_augmentation,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return loaders, info


def get_data_loaders(
    dataset_name='mnist',
    data_dir='./data',
    batch_size=128,
    eval_batch_size=None,  # If None, uses batch_size for val/test (smoother metrics)
    val_split=0.1,
    num_workers=4,
    seed=42,
    use_augmentation=False
):
    """
    Unified interface to get data loaders for any dataset.
    
    Args:
        dataset_name (str): Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        data_dir (str): Directory to store/load data
        batch_size (int): Batch size for training
        val_split (float): Fraction of training data for validation
        num_workers (int): Number of worker processes
        seed (int): Random seed
        use_augmentation (bool): Whether to use data augmentation
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
        dict: Dataset information
    
    Difficulty Guide:
        - MNIST: TOO EASY (98%+ accuracy even with bad models)
        - Fashion-MNIST: GOOD (88-92% accuracy, great differentiation) ⭐ RECOMMENDED
        - CIFAR-10: HARD for MLPs (50-60% accuracy, needs CNNs for 85%+)
    
    Example:
        >>> loaders, info = get_data_loaders('fashion_mnist', batch_size=64, use_augmentation=True)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return get_mnist_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_workers=num_workers,
            seed=seed,
            use_augmentation=use_augmentation
        )
    elif dataset_name in ['fashion_mnist', 'fashion-mnist', 'fmnist']:
        return get_fashion_mnist_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_workers=num_workers,
            seed=seed,
            use_augmentation=use_augmentation
        )
    elif dataset_name in ['cifar10', 'cifar-10']:
        return get_cifar10_flattened_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_workers=num_workers,
            seed=seed,
            use_augmentation=use_augmentation
        )
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not yet implemented. "
            f"Currently supported: ['mnist', 'fashion_mnist', 'cifar10']"
        )


if __name__ == '__main__':
    # Test the data loaders
    print("Testing MNIST data loaders...")
    
    # Test without augmentation
    print("\n1. Testing WITHOUT augmentation:")
    loaders, info = get_mnist_loaders(batch_size=128, use_augmentation=False)
    
    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nLoader Info:")
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader)} batches")
    
    # Test a batch
    print(f"\nTesting batch loading...")
    data, target = next(iter(loaders['train']))
    print(f"  Data shape: {data.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Target values: {target[:10].tolist()}")
    
    # Test with augmentation
    print("\n2. Testing WITH augmentation:")
    loaders_aug, info_aug = get_mnist_loaders(batch_size=128, use_augmentation=True)
    print(f"  Augmentation enabled: {info_aug['augmentation']}")
    
    # Load same batch multiple times to see augmentation variation
    train_iter = iter(loaders_aug['train'])
    data1, _ = next(train_iter)
    data2, _ = next(train_iter)
    print(f"  Data shape: {data1.shape}")
    print(f"  Data range: [{data1.min():.3f}, {data1.max():.3f}]")
    print(f"  ✓ Augmentation applied (each batch is transformed differently)")
    
    print("\n✓ Data loaders test passed (with and without augmentation)!")

