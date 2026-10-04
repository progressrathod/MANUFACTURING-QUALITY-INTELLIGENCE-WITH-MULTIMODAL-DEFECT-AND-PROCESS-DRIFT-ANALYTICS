"""Step 4 - PyTorch datasets, transforms and DataLoaders."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
from src.config import IMG_SPLIT, IMG_SIZE, BATCH_SIZE, IMG_CLASSES
 
 
def get_transforms(train: bool):
    steps = [transforms.Grayscale(num_output_channels=1),
             transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:                                     # augmentation only on training data
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(15)]
    steps += [transforms.ToTensor(),              # 0-255 -> 0-1
              transforms.Normalize([0.5], [0.5])]  # -> -1 to 1
    return transforms.Compose(steps)
 
 
def get_datasets():
    train_ds = datasets.ImageFolder(IMG_SPLIT / "train", get_transforms(True))
    val_ds = datasets.ImageFolder(IMG_SPLIT / "val", get_transforms(False))
    test_ds = datasets.ImageFolder(IMG_SPLIT / "test", get_transforms(False))
    # ImageFolder sorts folders A-Z: def_front -> 0, ok_front -> 1
    assert train_ds.classes == IMG_CLASSES, train_ds.classes
    return train_ds, val_ds, test_ds
 
 
def get_loaders():
    train_ds, val_ds, test_ds = get_datasets()
    # num_workers=0 avoids multiprocessing problems on Windows
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
 
 
if __name__ == "__main__":
    tr, va, te = get_loaders()
    x, y = next(iter(tr))
    print("Batch images:", x.shape, " labels:", y.shape)
    print("Train/Val/Test sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))
