from yolo import YOLO
from PIL import Image
import os



def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def detect_single_img(yolo, img_path):
    while True:
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

def detect_images_for_evaluate(yolo, imgs_path, batch_size):
    imgs = []
    img_files = os.listdir(imgs_path)
    for i in img_files:
        imgs.append(imgs_path + '\\' + i)
    yolo.detect_images_for_evaluate(imgs, batch_size)


def detect_images_in_folder(yolo, imgs_path, save_path):
    img_files = os.listdir(imgs_path)
    for i in img_files:
        img = os.path.join(imgs_path, i)
        save = os.path.join(save_path, i)
        image = Image.open(img)
        result_image = yolo.detect_image(image)
        result_image.save(save)

if __name__ == '__main__':
    # yolo = YOLO()
    # 检测单张图片
    #detect_single_img(YOLO(), '001.jpg')
    # detect_img(YOLO())

    # # # # # 检测多张图片
    imgs_path = 'E:\\.all-PythonCodes\\JPEGImages2'
    save_path = 'E:\\.all-PythonCodes\\Vis'
    detect_images_for_evaluate(YOLO(), imgs_path, 16)
    detect_images_in_folder(YOLO(), imgs_path, save_path)
