import numpy as np
import os
import tensorflow as tf
from PIL import Image
import modelcnn

tf.disable_v2_behavior()


def ZoomPicture(img):
    """
    计算原始图像缩放比例, 用来构建图像金字塔
    :param img: 原始图像
    :return: 图像缩放比例列表
    """
    pic_pro = 1.0  # 设置初始比例为1.0
    h, w, _ = img.shape  # 获取图像长宽
    # 按比例调整图像大小, 将最短边固定为500
    if min(w, h) > 500:
        pic_pro = 500.0 / min(h, w)
        w = int(w * pic_pro)
        h = int(h * pic_pro)
    elif max(w, h) < 500:
        pic_pro = 500.0 / max(h, w)
        w = int(w * pic_pro)
        h = int(h * pic_pro)

    pic_size = list()  # 用来保存图像缩放比例
    factor = 0.709  # 缩放因子
    factor_count = 0  # 缩放次数, 保留原始图片为第一张
    min_num = min(h, w)
    while min_num >= 12:  # 确保图像最短边不会小于12
        pic_size.append(pic_pro * pow(factor, factor_count))
        min_num *= factor
        factor_count += 1
    return pic_size


def rect2square(rectangles):
    """
    将长方形框转换为矩形, 在进入R-net和O-net时要进行缩放, 矩形框图像不易变形
    :param rectangles:原始框位置坐标
    :return: 正方形坐标
    """
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    L = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - L * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - L * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([L], 2, axis=0).T
    return rectangles


def NMS(rectangles, threshold):
    """
    非极大抑制
    :param rectangles: 所有预测框的位置
    :param threshold: 重合阈值
    :return: 剔除冗余后剩余框的位置
    """
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1, y1 = boxes[:, 0], boxes[:, 1]  # 左上角位置
    x2, y2 = boxes[:, 2], boxes[:, 3]  # 右下角位置
    s = boxes[:, 4]  # 框的得分
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())  # 给得分排序
    pick = list()
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)  # 交并比, 两矩形交集面积/两矩形最小集的面积
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


def facedetect_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    """
    对P-net的结果进行解码, 将预测结果映射到原始图像中
    :param cls_prob: 框中有人脸的概率
    :param roi: 框的位置
    :param out_side: 当前图像的长宽
    :param scale: 缩放比例
    :param width: 原始图像宽
    :param height: 原始图像高
    :param threshold: 置信度阈值
    :return: 非极大抑制后框的位置坐标及对应得分
    """
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    stride = 0  # 映射比例
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    (x, y) = np.where(cls_prob >= threshold)  # 获取满足阈值的网格点位置

    boundingbox = np.array([x, y]).T
    # 找到对应原图的位置
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)  # 框左上角点位置
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)  # 框右下角点位置
    # 输出框的点位置, 注意plt输出的图像左下角为零点, 对应OpenCV中左上角为零点
    # plt.scatter(bb1[:, 0], bb1[:,1], linewidths=1)
    # plt.scatter(bb2[:, 0], bb2[:,1], linewidths=1, c='r')
    # plt.show()
    boundingbox = np.concatenate((bb1, bb2), axis=1)

    # 左上角点的偏移坐标
    dx1, dx2 = roi[0][x, y], roi[1][x, y]
    # 右下角点的偏移坐标
    dx3, dx4 = roi[2][x, y], roi[3][x, y]

    score = np.array([cls_prob[x, y]]).T

    # 计算在原图上的真实坐标
    offset = np.array([dx1, dx2, dx3, dx4]).T
    boundingbox = boundingbox + offset * 12.0 * scale

    rectangles = np.concatenate((boundingbox, score), axis=1)  # 组合框的位置与得分
    rectangles = rect2square(rectangles)  # 调整框的形状, 方便后续操作
    location = list()
    for i in range(len(rectangles)):  # 比较调整后位置, 不能超出原图
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            location.append([x1, y1, x2, y2, sc])  # 保存调整后框的位置及得分信息
    return NMS(location, 0.3)


def facefilter_24net(cls_prob, roi, rectangles, width, height, threshold):
    """
    图像滤波, 筛选并调整高可信度人脸候选框
    :param cls_prob:可信度
    :param roi:如何调整候选框位置
    :param rectangles:候选框坐标
    :param width:原始图像宽
    :param height:原始图像高
    :param threshold:置信度阈值
    :return:非极大抑制后框的位置坐标及对应得分
    """
    # 通过阈值筛选可信度较高的部分
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    # 原始人脸框位置
    x1, y1 = rectangles[pick, 0], rectangles[pick, 1]
    x2, y2 = rectangles[pick, 2], rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    # 调整参数
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 人脸框的宽和高
    w = x2 - x1
    h = y2 - y1

    # 人脸框调整
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    # 组合框的位置和得分并调整为正方形
    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect2square(rectangles)
    location = list()
    for i in range(len(rectangles)):  # 调整所有框不能超出图像
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            location.append([x1, y1, x2, y2, sc])
    return NMS(location, 0.3)


def facefilter_48net(cls_prob, roi, local, rectangles, width, height, threshold):
    """
    最终筛选
    :param cls_prob:置信度
    :param roi:调整方式
    :param local:关键点位置
    :param rectangles:框的位置坐标
    :param width:原始图片宽
    :param height:原始图片高
    :param threshold:置信度阈值
    :return:非极大抑制后框的位置坐标及对应得分
    """
    # 剔除置信度过低的框
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    # 得到剩余框坐标
    x1, y1 = rectangles[pick, 0], rectangles[pick, 1]
    x2, y2 = rectangles[pick, 2], rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    # 5个人脸特征点
    pts0 = np.array([(w * local[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * local[pick, 5] + y1)[0]]).T

    pts2 = np.array([(w * local[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * local[pick, 6] + y1)[0]]).T

    pts4 = np.array([(w * local[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * local[pick, 7] + y1)[0]]).T

    pts6 = np.array([(w * local[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * local[pick, 8] + y1)[0]]).T

    pts8 = np.array([(w * local[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * local[pick, 9] + y1)[0]]).T

    # 调整人脸框的位置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)

    pick = list()
    for i in range(len(rectangles)):  # 调整所有框不能超出图像
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            # 组合人脸框位置及关键点位置
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)


def get_MER(x1, x2, y1, y2, img):
    """
    通过坐标裁剪对应图像
    :param x1: 左上横坐标
    :param x2: 右下横坐标
    :param y1: 左上纵坐标
    :param y2: 右下纵坐标
    :param img: 图片
    :return: 裁剪后图片
    """
    cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
    return cropped


def get_files(file_dir):
    """
    读取图片数据
    :param file_dir: 训练图片文件地址
    :return: 乱序后图片路径列表 乱序后标签对应列表
    """
    # 建立空列表
    openeyes = list()  # 存放睁眼图片路径地址
    label_open = list()  # 对应睁眼图片的标签
    closeeyes = list()  # 存放闭眼图片路径地址
    label_close = list()  # 对应闭眼图片的标签

    # 从file_dir路径下读取数据，存入空列表中
    for file in os.listdir(file_dir):  # file就是要读取的图片带后缀的文件名
        name = file.split(sep='.')  # 根据图片名称获取标签, 如处理后name为[open, 1, jpg]
        if name[0] == 'open':  # name[0]获取图片名
            openeyes.append(file_dir + file)
            label_open.append(0)  # 记0为睁眼, 1为闭眼
        else:
            closeeyes.append(file_dir + file)
            label_close.append(1)

    # print('There are %d open\nThere are %d close' % (len(open), len(close)))

    image_list = np.hstack((openeyes, closeeyes))  # 在水平方向平铺合成一个行向量
    label_list = np.hstack((label_open, label_close))
    temp = np.array([image_list, label_list])  # 生成一个数组，2行，每列图片和标签一一对应
    temp = temp.transpose()
    np.random.shuffle(temp)  # 乱序

    # 存储乱序后的图片及标签
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    图片格式转换及队列生成
    :param image: 要转换的图像
    :param label: 图像对应标签
    :param image_W: 图像要转换的宽
    :param image_H: 图像要转换的高
    :param batch_size: 一批次图像数量
    :param capacity: 队列最大容量
    :return: 转换后的图像队列及标签
    """
    """
    注意修改
    image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    image = tf.cast(image, tf.string)  # 将列表转换成tf能够识别的格式
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])  # 生成队列

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 解码得到图像对应的三维矩阵，注意类型需要转换，这里为uint8

    # 最近邻插值法调整图像大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)  # 类型转换
    image = tf.image.per_image_standardization(image)  # 图片标准化处理，加速神经网络的训练

    # 按顺序读取队列中的数据
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64, capacity=capacity)

    return image_batch, label_batch


def get_one_image(img_list):
    """
    随机选取测试集中一张图片
    :param img_list: 图片路径列表
    :return: 返回挑选图片的数组
    """
    n = len(img_list)
    ind = np.random.randint(0, n)
    img_dir = img_list[ind]

    image = Image.open(img_dir)
    image = image.resize([104, 104])
    image = np.array(image)  # 转成向量格式
    return image


def get_label(img):
    """
    判断图片为睁眼或是闭眼
    :param img: 输入图片
    :return: 标记
    """
    classname = ['open', 'close']
    image_array = np.array(img)

    with tf.Graph().as_default():
        BATCH_SIZE = 1  # 这里我们要输入的是一张图(预测这张随机图)
        N_CLASSES = 2  # 二分类

        image = tf.cast(image_array, tf.float32)  # 将列表转换成tf能够识别的格式
        image = tf.image.per_image_standardization(image)  # 图片标准化处理
        image = tf.reshape(image, [1, 104, 104, 3])
        logit = modelcnn.cnn_net(image, BATCH_SIZE, N_CLASSES)  # 神经网络输出层的预测结果
        logit = tf.nn.softmax(logit)  # 归一化处理

        x = tf.placeholder(tf.float32, shape=[104, 104, 3])  # x变量用于占位，输入的数据要满足这里定的shape

        # 训练好的模型路径
        logs_train_dir = './log/'

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            # 载入模型
            if ckpt and ckpt.model_checkpoint_path:  # checkpoint存在且其存放的变量不为空
                saver.restore(sess, ckpt.model_checkpoint_path)  # 当前会话中，恢复该路径下模型的所有参数（即调用训练好的模型）
            else:
                print('模型加载失败')

            # 通过saver.restore()恢复了训练模型的参数（即：神经网络中的权重值），这样logit才能得到想要的预测结果
            prediction = sess.run(logit, feed_dict={x: image_array})  # 输入随机抽取的那张图片数据，得到预测值
            max_index = np.argmax(prediction)  # 获取输出结果中最大概率的索引(下标)
            pre = classname[max_index]

    return pre

