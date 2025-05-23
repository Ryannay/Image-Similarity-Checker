from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [...]  # Your label loading logic here

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path)
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
