from abc import abstractmethod
import numpy as np
import torchvision
import traceback
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from img_vector_database import ImgVectorDatabase
from itertools import combinations
from utils import print_demo
from img_vectorizer import ImgVectorizer
class ImgDataHandler():
    def __init__(self, data_url="default", dim=256, model_size=18):
          self.vectorizer = ImgVectorizer(dim, model_size)
          self.dataset = ImgDataset(root_dir=data_url)
          self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
    def get_data(self):
          return self.vectorizer.vectorize(self.dataset)

class ImgDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_name)

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'file_name': self.image_files[idx]
        }

class MultiVectorFusion(torch.nn.Module):
    def __init__(self, dim=256, combination_num=2):
        super().__init__()
        self.dim = dim
        # for combinations feature
        self.combination_num = combination_num
        self.conv1d = torch.nn.Conv1d(self.dim, self.dim, kernel_size=self.combination_num, padding=0, bias=False)

        # for permute feature
        self.conv2d = torch.nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=8),
            num_layers=4,
        )
        self.fc = torch.nn.Linear(self.dim, 1)

    def combinations_feature(self, x):
        x = torch.nn.functional.adaptive_max_pool2d(x, 1).flatten(1)  # 4, 512
        x = torch.stack(
            [torch.stack(couple, dim=1) for couple in combinations(x, self.combination_num)])  # torch.Size([6, 512, 2])
        x = self.conv1d(x).flatten(1)
        logits = self.fc(x).mean()  # logits 为 6 种 feature 组合的结果，求均值
        return logits

    def permute_feature(self, x):
        x = self.conv2d(x)  # 4, 512, 1, 1
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, 512).unsqueeze(0)  # 1, 4, 512
        x = self.transformer(x)  # 1, 4, 512
        return self.fc(x[:, 0, :])

    def forward(self, x, type="p"):
        if type == "p":
            self.permute_feature(x)
        elif type == "c":
            self.combinations_feature(x)
        else:
            print("Error")
    # multi_images = torch.rand(4, 512, 1, 1)
    # model = MultiVectorFusion()
    # res = model(multi_images)
    # print(res)  # => tensor(1.0893, grad_fn=<MeanBackward0>)

def main():
    dataset_url = r"./data/dataset/"
    testset_url = r"./data/testset/"
    dim = 256
    model_size= 18
    collection_name = "Demo_18_256"

    dataBase = ImgVectorDatabase()
    dataBase.check_collection(collection_name)

    dataBase.add_collection(dim=dim,collection_name=collection_name)

    handler = ImgDataHandler(dataset_url, dim, model_size)
    handler2 = ImgDataHandler(testset_url, dim, model_size)

    img_vector, data_name = handler.get_data()
    img_vector = np.array(img_vector).reshape(-1, dim)

    test_vector, test_name = handler2.get_data()
    test_vector = np.array(test_vector).reshape(-1, dim)
    dataBase.add_entities(collection_name, img_vector, data_name)
    result = dataBase.query_entities(collection_name, test_vector)

    test_url_list = []
    for item in test_name:
            test_url_list.append(testset_url + item)
    res_url_list = []
    for hits in result:
        res_url = []
        for hit in hits:
            print(f"hit: {hit}, prime key: {hit.entity.get('pk')}\n")
            res_url.append(dataset_url + hit.entity.get('pk'))
        print("\n")
        res_url_list.append(res_url)
    print_demo(test_url_list, res_url_list)

if __name__ == "__main__":
    main()