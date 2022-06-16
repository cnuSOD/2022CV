import cv2
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input, MaxPool2D,
                          Permute, Reshape)
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import other_func


def create_Pnet(weight_path):
    inputs = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性。
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


def create_Rnet(weight_path):
    inputs = Input(shape=[24, 24, 3])
    # 24,24,3 -> 22,22,28 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 11,11,28 -> 9,9,48 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 128 -> 2
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    # 128 -> 4
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([inputs], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model


def create_Onet(weight_path):
    inputs = Input(shape=[48, 48, 3])
    # 48,48,3 -> 46,46,32 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 23,23,32 -> 21,21,64 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 3,3,128 -> 128,12,12
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 1152 -> 256
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 256 -> 2
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)  # 可信度
    # 256 -> 4
    bbox_regress = Dense(4, name='conv6-2')(x)  # 框调整方式
    # 256 -> 10
    landmark_regress = Dense(10, name='conv6-3')(x)  # 关键点位置

    model = Model([inputs], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)
    return model


class mtcnn:
    def __init__(self):
        # 使用模型创建三层网络
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    def detectFace(self, img, threshold):
        img_copy = (img - 127.5) / 127.5  # 图像归一化
        origin_h, origin_w, _ = img_copy.shape  # 获取图像长宽
        pic_size = other_func.ZoomPicture(img)  # 获取图像金字塔

        # P-net网络, 在图片中获取大量候选框
        out = list()
        for size in pic_size:
            hs = int(origin_h * size)
            ws = int(origin_w * size)
            img_pnet = cv2.resize(img_copy, (ws, hs))  # 根据缩放比例调整图片
            inputs = np.expand_dims(img_pnet, 0)
            # 将图片输入P-net进行预测
            output = self.Pnet.predict(inputs)
            output = [output[0][0], output[1][0]]  # 按照每张图片进行输出, 可以取消batch_size维度
            out.append(output)

        rectangles = list()
        #   取出图像金字塔中的每张图片的预测结果
        for i in range(len(pic_size)):
            cls_prob = out[i][0][:, :, 1]  # 框中有人脸的概率
            roi = out[i][1]  # 对应框的位置
            out_h, out_w = cls_prob.shape  # 当前图片的长宽
            out_side = max(out_h, out_w)
            # 将预测结果映射到原始图像中
            rectangle = other_func.facedetect_12net(cls_prob, roi, out_side, 1 / pic_size[i], origin_w, origin_h,
                                                    threshold[0])
            rectangles.extend(rectangle)

        rectangles = np.array(other_func.NMS(rectangles, 0.7))

        if len(rectangles) == 0:
            return rectangles

        # R-net网络, 进一步筛选候选框
        predict_24_batch = list()
        for rectangle in rectangles:
            # 根据P-net结果在原图上截取相应部分
            img_rnet1 = img_copy[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            img_rnet = cv2.resize(img_rnet1, (24, 24))  # 调整大小为24 x 24
            predict_24_batch.append(img_rnet)

        # 可信度, 如何调整图片对应的框
        cls_prob, roi_prob = self.Rnet.predict(np.array(predict_24_batch))
        rectangles = other_func.facefilter_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        # O-net网络, 精修人脸框
        predict_batch = list()
        for rectangle in rectangles:
            img_onet1 = img_copy[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            img_onet = cv2.resize(img_onet1, (48, 48))
            predict_batch.append(img_onet)

        # 置信度, 调整, 5点关键点位置
        cls_prob, roi_prob, local_prob = self.Onet.predict(np.array(predict_batch))
        rectangles = other_func.facefilter_48net(cls_prob, roi_prob, local_prob, rectangles, origin_w, origin_h,
                                                 threshold[2])

        return rectangles
