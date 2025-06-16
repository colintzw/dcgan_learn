import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = glob(os.path.join(root_dir, "*.png"))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # [H,W,C] -> [C,H,W], and to float [0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data/preprocessed"
)
ICON_DATASET = SimpleDataset(DATA_FOLDER)
