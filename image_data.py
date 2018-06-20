from PIL import Image
import numpy as np
import glob


def dr_img_dataset():
    arr = []
    images = glob.glob("diabetic_dataset/train/new/*jpeg")
    for image in images:
        pic = Image.open(image)
#         print("pic size:",pic.size)
        data = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)
        data = data[0:2560,0:1920,:]
#         data = np.array(pic.getdata()).reshape()

        arr.append(data)
    return arr