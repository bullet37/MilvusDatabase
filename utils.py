from PIL import Image
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
import cv2
from datetime import datetime, timedelta
import logging
import uuid

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((enums[key], key) for key in enums)
    enums['name'] = reverse
    return type('Enum', (), enums)

STAGE_STATE = enum(INIT='INIT', SUCCEED='SUCCEED', FAILED='FAILED', EXCEPTION='EXCEPTION')
HTTP_METHOD = enum(GET='GET', PUT='PUT', POST='POST', DELETE='DELETE')
class Model(object):
    def __init__(self):
        self.err_code = None
        self.state = constants.STAGE_STATE.INIT
    def pre_process(self):
        return constants.STAGE_STATE.SUCCEED, None

    def process(self):
        return constants.STAGE_STATE.SUCCEED, None

    def post_process(self):
        return constants.STAGE_STATE.SUCCEED, None

    def run(self):
        start_ts = time.time()
        logging.info('start processing: {}'.format(self.__class__))
        for func in [self.pre_process, self.process, self.post_process]:
            try:
                s_time = time.time()
                logging.info('start run func: {}'.format(func.__name__))

                self.state, message = func()
                message = str(message)
                self.err_code = message

                e_time = time.time()
                time_cost = e_time - s_time
                logging.info('finish run {}, time_cost: {}s, state: {}, message: {}'.format(
                    func.__name__, time_cost, self.state, message))

                if self.stop or self.state != constants.STAGE_STATE.SUCCEED:
                    return self.state, message
            except Exception:
                err_msg = traceback.format_exc()
                logging.error('failed to run task: {}, err: {}'.format(
                    func.__name__, err_msg))
                self.state = constants.STAGE_STATE.FAILED
                self.err_code = -1
                return self.state, self.err_code
        end_ts = time.time()
        time_cost = end_ts - start_ts
        logging.info('finish processing {}, time_cost: {}s, state: {}, message: {}'.format(
            self.__class__, time_cost, self.state, self.err_code
        ))
        return self.state, self.err_code

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

def print_time(duration):
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    logging.info(f"Run time: {hours} hour, {minutes} minutes, {seconds} seconds")

def print_demo(query_paths, result_paths):
    plt.figure(figsize=(15, 6))
    for i in range(len(query_paths)):
        plt.subplot(2, 9, i * 9 + 1)
        query_image = Image.open(query_paths[i])
        plt.imshow(query_image)
        plt.title(f'Query {i + 1}')
        plt.axis('off')

        for j in range(len(result_paths[i])):
            plt.subplot(2, 9, i * 9 + j + 2)
            result_image = Image.open(result_paths[i][j])
            plt.imshow(result_image)
            plt.title(f'Result {j + 1}')
            plt.axis('off')
    plt.show()

def img_to_nparray(image, resize=True, img_size=(256,256)):
    # Check if the image is a PIL Image
    if isinstance(image, Image.Image):
        if resize:
            image = image.resize(img_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

    # Check if the image is a numpy array (read by cv2)
    elif isinstance(image, np.ndarray):
        # Convert from BGR to RGB if it's a color image with 3 channels (OpenCV uses BGR format)
        if resize:
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
        if image.ndim == 3 and image.shape[2] == 3:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_np = image
    else:
        raise TypeError("Unsupported image type, expected PIL.Image.Image or numpy.ndarray")

    # Convert the numpy array to a list and back to a numpy array (this step seems redundant and could be skipped)
    image_list = image_np.tolist()
    image_np = np.array(image_list)
    image_np = np.transpose(image_np, (2, 0, 1))  # [256,256,3] => [3,256,256]
    return image_np

def img_to_tensor(image, resize=True, img_shape = [1,3,256,256]):
    image_np = img_to_nparray(image, resize)
    # Convert the numpy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0)
    return image_tensor


def get_uuid():
    return str(uuid.uuid4())

def change2root(func):
    def wrapper(*args, **kwargs):
        original_cwd = os.getcwd()
        os.environ["root"] = sys.path[1]
        os.chdir(os.environ["root"])
        result = func(*args, **kwargs)  # 调用原始函数
        os.chdir(original_cwd)
        return result
    return wrapper # usage: @change2root