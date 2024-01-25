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
from vectorizers.res_vectorizer import ResVectorizer

class ImgDataHandler():
    def __init__(self, vectorizer, data_url="default", dim=256, model_size=18):
          self.vectorizer = vectorizer
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


def main():
    # if len(sys.argv) < 2 or len(sys.argv) > 3:
    #     print("Usage: %s image.jpg [dir]" % sys.argv[0])
    # else:
    #     image_path, wd = sys.argv[1], '.' if len(sys.argv) < 3 else sys.argv[2]

    dataset_url = r"./data/dataset/"
    testset_url = r"./data/testset/"
    dim = 256
    model_size= 50
    collection_name = "Demo_18_256"

    dataBase = ImgVectorDatabase()
    exist_flag = dataBase.check_collection(collection_name)
    if exist_flag is False:
        entities_num = database.add_collection(dim=dim, collection_name=collection_name)


    vectorizer = ResVectorizer(dim, model_size)
    handler = ImgDataHandler(vectorizer, dataset_url, dim, model_size)
    handler2 = ImgDataHandler(vectorizer, testset_url, dim, model_size)

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