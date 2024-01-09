from abc import abstractmethod
import torchvision
import traceback
import torch
from datetime import datetime, timedelta
from torch.utils.data import Subset
import logging

class Vectorizer(object):
    def __init__(self):
        pass
    @abstractmethod
    def vectorize_data(self, data):
        pass
    @abstractmethod
    def vectorize_dataset(self, dataset):
        pass

    def callback_vectorize_data(self, data):
        try:
            self.vectorize_data(data)
        except:
            logging.error("callback vectorize data error: " + str(traceback.format_exc()))

    def callback_vectorize_dataset(self, dataset):
        try:
            self.vectorize_dataset(dataset)
        except:
            logging.error("callback vectorize dataset error: " + str(traceback.format_exc()))


class ImgVectorizer(Vectorizer):
    def __init__(self, dim=256, model_size=18, chunk_size=1000, device="cpu"):
        Vectorizer.__init__(self)
        self.dim = dim
        self.chunk_size = chunk_size
        self.model_size = model_size
        self.device = device
        self.model = None
        # summary(self.model)
        if self.model_size == 50:
            self.model = torchvision.models.resnet50(pretrained=True).to(device)
        elif self.model_size == 34:
            self.model = torchvision.models.resnet34(pretrained=True).to(device)
        elif self.model_size == 18:
            self.model = torchvision.models.resnet18(pretrained=True).to(device)

    def vectorize_data(self, img):
        img = img.view(1, 3, 256, 256).to(self.device)
        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if self.model_size == 18 or self.model_size == 34:
            x = self.model.layer1(x)  # 64
            if self.dim >= 128:
                x = self.model.layer2(x)  # 128
                if self.dim >= 256:
                    x = self.model.layer3(x)  # 256
                    if self.dim == 512:
                        x = self.model.layer4(x)  # 512
        elif self.model_size == 50:
            x = self.model.layer1(x)  # 256
            if self.dim >= 512:
                x = self.model.layer2(x)  # 512
                if self.dim >= 1024:
                    x = self.model.layer3(x)  # 1024
                    if self.dim == 2048:
                        pass
                        # x = model.layer4(x) # 2048
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1) .detach().numpy()
        return x

    def vectorize_dataset(self, dataset):
        vector_list, name_list = [], []
        for i in range(len(dataset)):
            try:
                img = self.vectorize_data(dataset[i]['image'])
                name_list.append(dataset[i]['file_name'])
                vector_list.append(img)
            except:
                logging.error("vectorize_dataset error:: " + str(traceback.format_exc()))
        return vector_list, name_list

    def vectorize(self, dataset):
        vector_list, name_list = [], []
        start_time = datetime.now()
        dataset_length = len(dataset)
        if dataset_length < self.chunk_size:
            vector_list_TMP, name_list_TMP = self.vectorize_dataset(dataset)
            vector_list.extend(vector_list_TMP)
            name_list.extend(name_list_TMP)
        else:
            chunk_size = self.chunk_size
            pre = start_time
            for i in range(0, dataset_length, chunk_size):
                indices = range(i, i + chunk_size)
                subset = Subset(dataset, indices)
                vector_list_TMP, name_list_TMP = self.vectorize_dataset(subset)

                vector_list.extend(vector_list_TMP)
                name_list.extend(name_list_TMP)
                now = datetime.now()
                print(f"Vectorized {i}th to {i + chunk_size} images, used time:{(now - pre).total_seconds()}")
                pre = now
        end_time = datetime.now()
        print(f"Vectorized entities used time: {(end_time - start_time).total_seconds()}")
        return vector_list, name_list