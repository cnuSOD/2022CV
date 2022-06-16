import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import other_func
import cv2
import modelcnn
import numpy as np

tf.disable_v2_behavior()


def checkcnn(test_dir):
    """
    判断cnn网络的图片判断正确率
    :param test_dir: 测试集路径
    :return: 正确率和判断错误列表
    """
    test_img, test_label = other_func.get_files(test_dir)
    print(len(test_img))
    print(len(test_label))
    label_falses = list()  # 预测错误标签列表
    label_false = dict()
    for img_dir, i in zip(test_img, range(len(test_img))):
        image = Image.open(img_dir)
        image = np.array(image)  # 转成向量格式
        image_re = cv2.resize(image, dsize=(104, 104), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(cv2.cvtColor(image_re, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        simple = other_func.get_label(image_rgb)  # 预测
        if simple == "open":
            simple = 0
        else:
            simple = 1

        if simple != test_label[i]:
            label_false = {"dir": img_dir,
                           "forecast": simple,
                           "real": test_label[i]}
            label_falses.append(label_false)
        print("已完成第{}张".format(i + 1))
    # 准确率计算
    pre = (len(test_img)-len(label_falses)) / len(test_img)
    return pre, label_falses


if __name__ == '__main__':
    # 计算正确率
    file = "./data/test/"
    rate, label_list = checkcnn(file)
    print(label_list)
    print(len(label_list))
    print(rate)
