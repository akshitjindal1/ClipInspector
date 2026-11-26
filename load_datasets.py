import os
import random
import json
import logging
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Dataset directory, image subdirectory, and split_zhou filename suffix.
DATASET_LAYOUTS = {
    "caltech101": ("caltech-101", "101_ObjectCategories", "Caltech101"),
    "dtd": ("dtd", "images", "DescribableTextures"),
    "ucf101": ("ucf101", "UCF-101-midframes", "UCF101"),
    "oxford_pets": ("oxford_pets", "images", "OxfordPets"),
    "food101": ("food-101", "images", "Food101"),
}


class Datum:
    """Data instance for convenience."""
    def __init__(self, impath="", label=0, classname=""):
        self.impath = impath
        self.label = label
        self.classname = classname


class DatumDataset(Dataset):
    """Dataset created from Datum objects."""
    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform
    
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = Image.open(item.impath).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return (img, item.label)


def read_json(fpath):
    """Helper function to read a JSON file."""
    with open(fpath, 'r') as f:
        return json.load(f)


def get_clip_transform(image_size=224):
    """Return CLIP's standard transform."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def split_data_by_class(items, train_ratio=0.6, val_ratio=0.2):
    """Split data into train, val, test sets while maintaining class balance."""
    train, val, test = [], [], []
    
    # Find all unique class labels
    class_labels = set(item.label for item in items)
    
    for class_label in sorted(class_labels):
        class_items = [item for item in items if item.label == class_label]
        random.shuffle(class_items)
        n_total = len(class_items)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        train.extend(class_items[:n_train])
        val.extend(class_items[n_train:n_train + n_val])
        test.extend(class_items[n_train + n_val:])
    
    return train, val, test


def group_data_by_class(data_items):
    """Group data items by their class label."""
    by_class = {}
    for item in data_items:
        label = item.label
        if label not in by_class:
            by_class[label] = []
        by_class[label].append(item)
    return by_class


def sample_few_shot_data(items, num_shots, max_val_shots=4):
    """Sample data for few-shot learning."""
    by_class = group_data_by_class(items)
    
    sampled_items = []
    for label, class_items in by_class.items():
        random.shuffle(class_items)
        sampled_items.extend(class_items[:num_shots])
    
    return sampled_items


def filter_and_relabel_data(data, selected_labels, label_map):
    """Filter data to include only selected labels and relabel according to the map."""
    filtered = []
    for item in data:
        if item.label in selected_labels:
            filtered.append(
                Datum(impath=item.impath, 
                     label=label_map[item.label], 
                     classname=item.classname)
            )
    return filtered


def load_image_folder_dataset(image_dir, transform=None, dataset_name=None):
    """Load a dataset based on directory structure."""
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    
    # Special handling for specific datasets
    if dataset_name and dataset_name.lower() == "caltech101":
        # Filter out specific folders
        ignored_folders = ["BACKGROUND_Google", "Faces_easy"]
        valid_folders = [d for d in os.listdir(image_dir) 
                         if os.path.isdir(os.path.join(image_dir, d)) 
                         and d not in ignored_folders]
        
        # Class name replacements
        name_replacements = {
            "airplanes": "airplane",
            "Faces": "face",
            "Leopards": "leopard",
            "Motorbikes": "motorbike",
        }
        
        # Create items manually
        items = []
        for idx, folder in enumerate(sorted(valid_folders)):
            folder_path = os.path.join(image_dir, folder)
            classname = name_replacements.get(folder, folder.replace("_", " "))
            
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    items.append(Datum(impath=file_path, label=idx, classname=classname))
        
        # Create classnames list
        classnames = [folder.replace("_", " ") if folder not in name_replacements else name_replacements[folder] 
                      for folder in sorted(valid_folders)]
        
    else:
        # Default handling
        classnames = [c.replace("_", " ") for c in dataset.classes]
        
        # Convert to Datum objects
        items = []
        for idx, (path, label) in enumerate(dataset.samples):
            items.append(Datum(impath=path, label=label, classname=classnames[label]))
    
    return items, classnames


def load_dataset(config, shots=None, subsample="all"):
    """
    Load dataset and create train, validation, and test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader, classnames, class_to_idx
    """
    dataset_name = config['dataset_name']
    root_path = config['data_root']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    image_size = config['image_size']
    seed = config['seed']
    
    # Get shots and subsample from config if not provided
    if shots is None:
        shots = config.get('shots', None)
    if subsample == "all":
        subsample = config.get('subsample', "all")
    
    # Set up transform
    transform = get_clip_transform(image_size)
    
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_LAYOUTS:
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Supported datasets: {', '.join(DATASET_LAYOUTS)}")
    directory, image_subdir, split_name = DATASET_LAYOUTS[dataset_key]
    dataset_dir = os.path.join(root_path, directory)
    image_dir = os.path.join(dataset_dir, image_subdir)
    split_path = os.path.join(dataset_dir, f"split_zhou_{split_name}.json")
    os.makedirs(os.path.join(dataset_dir, "split_fewshot"), exist_ok=True)
    random.seed(seed)

    # Load train/val/test split from JSON if available
    if os.path.exists(split_path):
        logging.info(f"Loading split from {split_path}")
        split_data = read_json(split_path)

        # Convert JSON data to Datum objects
        train, val, test = [], [], []

        for impath, label, classname in split_data.get("train", []):
            impath = os.path.join(image_dir, impath)
            train.append(Datum(impath=impath, label=int(label), classname=classname))

        for impath, label, classname in split_data.get("val", []):
            impath = os.path.join(image_dir, impath)
            val.append(Datum(impath=impath, label=int(label), classname=classname))

        for impath, label, classname in split_data.get("test", []):
            impath = os.path.join(image_dir, impath)
            test.append(Datum(impath=impath, label=int(label), classname=classname))
    else:
        # If no split file, load from directory structure
        logging.info(f"No split file found at {split_path}, loading from directory structure")
        items, classnames = load_image_folder_dataset(image_dir, transform, dataset_name)
        train, val, test = split_data_by_class(items)

    # Handle few-shot learning if shots is specified
    if shots is not None and shots > 0:
        random.seed(seed)

        logging.info(f"Creating {shots}-shot dataset")
        
        # Sample few shots per class for training
        few_shot_train = sample_few_shot_data(train, shots)
        
        # Sample up to min(shots, 4) samples per class for validation
        # Group by class
        val_by_class = group_data_by_class(val)
        
        # Sample from each class
        few_shot_val = []
        for label, items in val_by_class.items():
            random.shuffle(items)
            few_shot_val.extend(items[:min(shots, 4)])
        
        train, val = few_shot_train, few_shot_val
    
    logging.info("CREATING THE SUBSAMPLE SPLIT")
    # Handle class subsampling
    if subsample != "all":
        # Get all class labels
        labels = set()
        for item in train:
            labels.add(item.label)
        labels = sorted(list(labels))
        
        # Split into base and new classes
        n = len(labels)
        m = n // 2  # half-point
        
        if subsample == "base":
            selected = labels[:m]  # first half
        else:  # "new"
            selected = labels[m:]  # second half
            
        # Create mapping from old to new labels
        label_map = {old: new for new, old in enumerate(selected)}
        
        # Filter and relabel data
        train = filter_and_relabel_data(train, selected, label_map)
        val = filter_and_relabel_data(val, selected, label_map)
        test = filter_and_relabel_data(test, selected, label_map)
    
    # Create datasets
    train_dataset = DatumDataset(train, transform=transform)
    val_dataset = DatumDataset(val, transform=transform)
    test_dataset = DatumDataset(test, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Extract classnames and class_to_idx
    label_to_classname = {}
    for item in test:
        if item.label not in label_to_classname:
            label_to_classname[item.label] = item.classname
    
    classnames = [label_to_classname[i] for i in range(len(label_to_classname))]
    
    class_to_idx = {name: i for i, name in enumerate(classnames)}
    
    logging.info(f"Dataset {dataset_name} loaded:")
    logging.info(f"  Train samples: {len(train_dataset)}")
    logging.info(f"  Val samples: {len(val_dataset)}")
    logging.info(f"  Test samples: {len(test_dataset)}")
    logging.info(f"  Classes: {len(classnames)}")
    
    return train_loader, val_loader, test_loader, classnames, class_to_idx
