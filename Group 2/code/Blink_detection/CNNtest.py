import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import other_func
import modelcnn
import numpy as np

tf.disable_v2_behavior()


def evaluate_one_image():
    # 修改成自己测试集的文件夹路径
    test_dir = './data/test/'

    test_img = other_func.get_files(test_dir)[0]
    image_array = other_func.get_one_image(test_img)

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
            # print("从指定路径中加载模型...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            # 载入模型
            if ckpt and ckpt.model_checkpoint_path:  # checkpoint存在且其存放的变量不为空
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  # 通过切割获取ckpt变量中的步长
                saver.restore(sess, ckpt.model_checkpoint_path)  # 当前会话中，恢复该路径下模型的所有参数（即调用训练好的模型）
                print('模型加载成功, 训练的步数为： %s' % global_step)
            else:
                print('模型加载失败')

            # 通过saver.restore()恢复了训练模型的参数（即：神经网络中的权重值），这样logit才能得到想要的预测结果
            prediction = sess.run(logit, feed_dict={x: image_array})  # 输入随机抽取的那张图片数据，得到预测值
            max_index = np.argmax(prediction)  # 获取输出结果中最大概率的索引(下标)
            if max_index == 0:
                pre = prediction[:, 0][0] * 100
                print('图片是睁眼的概率为： {:.2f}%'.format(pre))  # 下标为0，则为睁眼图片
            else:
                pre = prediction[:, 1][0] * 100
                print('图片是闭眼的概率为： {:.2f}%'.format(pre))  # 下标为1，则为闭眼图片

    plt.imshow(image_array)  # 接受图片并处理
    plt.show()  # 显示图片


if __name__ == '__main__':
    # 调用方法，开始测试
    evaluate_one_image()
