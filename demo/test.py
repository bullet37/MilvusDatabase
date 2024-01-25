# # testing for the function
# image = Image.open('./1.jpg')
# data = img_to_tensor(image)
# controller = Controller()
# print(controller.vectorizer_dict['yolo_vectorizer'].vectorize_data(data))

import gradio as gr
import numpy as np
from controller import Controller
from img_vector_database import ImgVectorDatabase
from PIL import Image
from utils import img_to_tensor
import sys, os
print(os.path.split(os.path.abspath(__file__)))
def img_vector_query(img_url, model, vector_length, top_k):
    dataset_url = r"D:\dataset/"
    controller = Controller()
    image = Image.open(img_url)
    tensor = img_to_tensor(image)

    if model=="YoloP":
        vectorizer_name = 'yolo_vectorizer'
        database_name = vectorizer_name + "__" + str(vector_length)

    elif model=="ResNet":
        vectorizer_name = 'res_vectorizer'
        model_size = 18
        database_name = vectorizer_name + "_18_" + str(vector_length)

    elif model=="Hasher":
        vectorizer_name = 'hash_vectorizer'
        database_name = vectorizer_name + "_" + str(vector_length)


    vectorizer = controller.vectorizer_cls_dict[vectorizer_name](dim=int(vector_length))
    test_vector = vectorizer.vectorize_data(tensor)

    dataBase = ImgVectorDatabase()

    result = dataBase.query_entities(database_name, test_vector, top_k)
    res_url = []
    for hits in result:
        for hit in hits:
            res_url.append(dataset_url+hit.entity.get('url'))
    # return image_urls # ,query_url
    return res_url

# https://zhuanlan.zhihu.com/p/624712372?utm_id=0&wd=&eqid=d4999b5400042f17000000036479aff6
demo = gr.Interface(
    description="Vector Database Demo",
    inputs=[
            gr.Image(height=200, width=200, show_label=True,label="upload_image",type="filepath"),
            gr.Radio(["ResNet", "YoloP", "Hasher"],show_label=True,value = "ResNet", label="Vectorize Model"),
            gr.Radio( choices=[128, 256, 512, 1024, 2048], value=128, label="vector dimension"),
            gr.Slider(minimum=0, maximum=100,value=8,label="search_top_K_img"),
            ],

    fn=img_vector_query,
    outputs=[gr.Gallery(),],
    #live=True,
    #css=css_code,
    article="",
)
if __name__ == "__main__":
    demo.queue()
    app, local_url, share_url = demo.launch(share=True)  # public link
    print(app, local_url, share_url)