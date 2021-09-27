from utils import convert_to_valid_path
from tqdm import tqdm
import torch
import sys
sys.path.append("../..")

import argparse
import os
import pickle
from glob import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from reality_check.dataset import transform
from reality_check.model import Network


class FakeCarsDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.files = glob("3D cars dataset/images/*.png")
        self.files = sorted(self.files, key=lambda x: int(x.split("/")[-1].split(".")[0]))

        assert len(self.files) == 100000

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')

        if self.transform != None:
            img = transform(img)

        return {"data": img}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, help="Loss name")
    parser.add_argument("--fold", type=int, help="Fold number")
    config = vars(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dataset = FakeCarsDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

    model = Network(config["loss"], dataset="Cars196", fold=config["fold"]).eval().to(device)

    embeddings = []
    for batch in tqdm(dataloader):
        data = batch["data"].to(device)

        embedding = model(data).detach().cpu()
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)

    os.makedirs("representations", exist_ok=True)
    with open(f"representations/{convert_to_valid_path(config['loss'])}_{config['fold']}.p", "wb") as f:
        pickle.dump(embeddings, f)
