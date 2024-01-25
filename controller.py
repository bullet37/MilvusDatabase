import logging
import importlib
import traceback
import time
import sys
from PIL import Image
import os
from utils import img_to_tensor
ROOT_SPACE = os.path.split(os.path.abspath(__file__))[0]

def get_class(module_name: str, class_name: str):
    try:
        module_object = importlib.import_module(module_name)  # 将模块加载为对象
        # print('module_name:', module_name, ' class_name:', class_name)
        module_object_cls = getattr(module_object, class_name)  # 获取类对象
        return module_object_cls
    except:
        logging.error("get_module_class_object error: " + str(traceback.format_exc()))
def get_class_object(module_name: str, class_name: str):
    try:
        module_object = importlib.import_module(module_name)  # 将模块加载为对象
        # print('module_name:', module_name, ' class_name:', class_name)
        module_object_cls = getattr(module_object, class_name)  # 获取类对象
        module_class_object = module_object_cls()  # 将类实例化
        return module_class_object
    except:
        logging.error("get_module_class_object error: " + str(traceback.format_exc()))

def get_class_name(module_name: str):
    # xxx_yyy => xxxYyy
    module_name_split = module_name.split('_')
    class_name = ''
    for module_split in module_name_split:
        module_name_upper = module_split[0].upper() + module_split[1:len(module_split)]
        class_name += module_name_upper
    return class_name
class Controller():
    def __init__(self):
        self.vectorizer_obj_dict = dict()
        self.vectorizer_cls_dict = dict()
        self.init_module()
        self.database_dict = dict()

    def init_module(self):
        vectorizer_path = ROOT_SPACE + r'/vectorizers'
        sys.path.append(vectorizer_path)
        vectorizer_list = os.listdir(vectorizer_path)
        print(vectorizer_list)
        for file_name in vectorizer_list:
            module_name = file_name.split('.')[0]
            if not str(file_name).endswith(r'_vectorizer.py'): continue
            try:
                class_name = get_class_name(module_name)
                class_cls = get_class(module_name, class_name)
                #class_obj = get_class_object(module_name, class_name)
                # if class_obj is None:
                #     logging.error(f'module %s is illegality' %file_name)
                #     continue
                #self.vectorizer_obj_dict[module_name] = class_obj  # 模块类
                self.vectorizer_cls_dict[module_name] = class_cls # 模块类
            except:
                logging.error('init_module error:{}'.format(traceback.format_exc()))

