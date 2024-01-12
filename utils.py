from PIL import Image
import matplotlib.pyplot as plt
import logging
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
