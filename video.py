# -------------------------------------#
#       调用摄像头检测
# -------------------------------------#
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2

yolo = YOLO()
# 调用摄像头
# vid = cv2.VideoCapture(0)  # capture=cv2.VideoCapture("1.mp4")
vid = cv2.VideoCapture("./img.mp4")  # 调用本地视频
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_path = "./200.mp4"
isOutput = True if output_path != "" else False
if isOutput:
    # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)  # 把视频存到3.mp4输出
    # out = cv2.VideoWriter("E:\\keras_yolov3_mobilenet-2000\\3.mp4", video_FourCC, video_fps, video_size) #把视频存到3.mp4输出

while (True):
    # 读取某一帧
    ref, frame = vid.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(yolo.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("video", frame)
    c = cv2.waitKey(30) & 0xff
    if isOutput:
        out.write(frame)
    if c == 27:
        vid.release()
        break

yolo.close_session()
