import torchvision
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from torch.utils.data import Subset
import logging
import os, sys
from vectorizers.vectorizer_base import Vectorizer

ROOT_SPACE = os.path.split(os.path.abspath(__file__))[0]
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

class PartialModel(nn.Module):
    def __init__(self):
        super(PartialModel, self).__init__()
        full_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        self.from_list = []
        for i, block in enumerate(full_model.model):
            self.from_list.append(block.from_)
        # 共42层
        self.model = nn.Sequential(full_model.model[0], full_model.model[1], full_model.model[2], full_model.model[3],
                                   full_model.model[4], full_model.model[5], full_model.model[6], full_model.model[7],
                                   full_model.model[8], full_model.model[9], full_model.model[10], full_model.model[11],
                                   full_model.model[12], full_model.model[13], full_model.model[14], full_model.model[15],
                                   full_model.model[16], full_model.model[17], full_model.model[18], full_model.model[19],
                                   full_model.model[16], full_model.model[17], full_model.model[18], full_model.model[19],
                                   full_model.model[20], full_model.model[21], full_model.model[22], full_model.model[23],)
        #self.seg_out_idx = [33, 42]
        self.detector_index = 23
        self.save = [4, 6, 10, 14, 16, 16, 17, 20, 23]

    def forward(self, x, len=20):
        cache = []
        for i, block in enumerate(self.model):
            if i >= len: break
            if self.from_list[i] != -1:
                x = cache[self.from_list[i]] if isinstance(self.from_list[i], int) \
                    else [x if j == -1 else cache[j] for j in self.from_list[i]]       #calculate concat detect
            x = block(x)
            cache.append(x if i in self.save else None)
        x = x.view(-1)[:512]
        return x

class YoloVectorizer(Vectorizer):
    def __init__(self, dim=256, chunk_size=1000, device="cpu"):
        Vectorizer.__init__(self)
        self.dim = dim
        self.chunk_size = chunk_size
        self.device = device
        self.model = PartialModel().eval()

    def vectorize_data(self, data):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif isinstance(data, list):
                data = torch.tensor(data)
            else:
                raise TypeError("Unsupported data type, expected list or numpy.ndarray")
        data = data.view(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            res = self.model(data)
        return res

    def vectorize_dataset(self, dataset):
        vector_list, name_list = [], []
        for i in range(len(dataset)):
            try:
                img = self.vectorize_data(dataset[i]['image'])
                name_list.append(dataset[i]['file_name'])
                vector_list.append(img)
            except:
                logging.error('vectorize_dataset error:{}'.format(traceback.format_exc()))
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
                end_point = min(i + chunk_size, dataset_length)
                indices = range(i, end_point)
                subset = Subset(dataset, indices)
                vector_list_TMP, name_list_TMP = self.vectorize_dataset(subset)

                vector_list.extend(vector_list_TMP)
                name_list.extend(name_list_TMP)
                now = datetime.now()
                logging.info(f"Vectorized {i}th to {i + chunk_size} images, used time:{(now - pre).total_seconds()}")
                pre = now
        end_time = datetime.now()
        logging.info(f"Vectorized entities used time: {(end_time - start_time).total_seconds()}")
        return vector_list, name_list
