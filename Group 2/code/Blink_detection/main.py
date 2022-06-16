import cv2
import math
import other_func
from model import mtcnn

file_list = list()  # 存储处理好的图片, 生成演示视频用

blink = 0  # 眨眼次数

model = mtcnn()  # 创建网络
threshold = [0.5, 0.6, 0.7]  # 检测阈值

cap = cv2.VideoCapture('video1.mp4')
sign = "open"
while True:
    flag, frame = cap.read()  # 按帧读取图片
    # 视频播放完成退出循环
    if not flag:
        break

    img = frame
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rectangles = model.detectFace(temp_img, threshold)  # 检测图片

    for rectangle in rectangles:
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])

        left_eye = rectangle[5:7]
        # print(left_eye)
        right_eye = rectangle[7:9]
        # print(right_eye)
        # 根据“三庭五眼”确定眼睛框选位置
        arc = math.atan(abs(right_eye[1] - left_eye[1]) / abs(right_eye[0] - left_eye[0]))
        W_eye = abs(right_eye[0] - left_eye[0]) / (2 * math.cos(arc))
        H_eye = W_eye / 2

        # 左眼
        x1, y1 = int(left_eye[0] - W_eye / 2), int(left_eye[1] - H_eye / 2)
        x2, y2 = int(left_eye[0] + W_eye / 2), int(left_eye[1] + H_eye / 2)
        left = other_func.get_MER(x1, x2, y1, y2, img)
        # cv2.imshow("left", left)
        left_re = cv2.resize(left, dsize=(104, 104), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        left_rgb = cv2.cvtColor(cv2.cvtColor(left_re, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        left_state = other_func.get_label(left_rgb)  # 判断睁闭眼情况
        # print(left_state)
        if left_state == "close" and sign == "open":
            blink += 1
            cv2.putText(img, "blink:{}".format(blink), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, 8)
            sign = left_state
        else:
            cv2.putText(img, "blink:{}".format(blink), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, 8)
            sign = left_state

        # 右眼
        rx1, ry1 = int(right_eye[0] - W_eye / 2), int(right_eye[1] - H_eye / 2)
        rx2, ry2 = int(right_eye[0] + W_eye / 2), int(right_eye[1] + H_eye / 2)
        # right = other_func.get_MER(rx1, rx2, ry1, ry2, img)
        # cv2.imshow("right", right)


        # 标记人脸
        cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                      (0, 0, 255), 2)
        cv2.putText(img, "blink:{}".format(blink), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, 8)

        # 框选眼睛
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

    img_res = img
    cv2.imshow("test", img)
    # 生成演示视频文件用
    img_res = cv2.resize(img_res, (1280, 720))
    file_list.append(img_res)

    # 按空格键直接结束识别
    if ord(' ') == cv2.waitKey(10):
        break

# 生成演示视频
fps = 24
video = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (1280, 720))
for item in file_list:
    video.write(item)
video.release()

# 关闭OpenCV窗口
cv2.destroyAllWindows()
cap.release()
