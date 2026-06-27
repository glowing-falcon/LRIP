import torchvision
from torch.utils.data import ConcatDataset, Dataset


class ConcatWithClasses(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

        # Check all classes are the same
        first_classes = datasets[0].classes
        for ds in datasets[1:]:
            if ds.classes != first_classes:
                raise ValueError(
                    "Datasets to concatenate have different class lists. "
                    f"First dataset classes: {first_classes}, "
                    f"Current dataset classes: {ds.classes}"
                )
        self.classes = first_classes


class ImageFolderDict(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds
        self.classes = getattr(base_ds, "classes", None)  # Needed?

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        image, label = self.base_ds[idx]
        return {"image": image, "label": label}


class MNISTInMemory(Dataset):
    def __init__(self, base_ds: Dataset):
        self.data = base_ds.data.unsqueeze(1)
        self.classes = base_ds.classes
        self.targets = base_ds.targets
        assert base_ds.transform is None, "Transform is not supported."
        self.data = self.data.float() / 255.0
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(self.data.mean(), self.data.std()),
        ])
        self.data = transform(self.data)
        if base_ds.target_transform is not None:
            self.targets = self.target_transform(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
