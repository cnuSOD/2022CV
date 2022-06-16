import tensorflow as tf

tf.disable_v2_behavior()


def cnn_net(images, batch_size, n_classes):
    """
    构建CNN网络用于识别眨眼情况
    :param image: 队列中的图片
    :param batch_size: 批次大小
    :param n_classes: 类别数量
    :return: 类别的概率
    """
    # 第一层卷积, 卷积核大小为3*3
    with tf.variable_scope('conv1') as scope:
        # 权重
        weights = tf.get_variable("weights", shape=[3, 3, 3, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # 偏置项
        biases = tf.get_variable('biases', shape=[16], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)  # 加入偏差
        conv1 = tf.nn.relu(pre_activation, name='conv1')  # 激活函数为relu

        # 第一层池化, 采用最大池化法
    with tf.variable_scope('pooling1_lrn') as scope:
        # 对conv1池化得到feature map
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        # lrn()：局部响应归一化, 防止过拟合
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # 第二层卷积, 卷积核大小为3*3
    with tf.variable_scope('conv2') as scope:
        # shape的第三位数字16等于上一层的tensor维度
        weights = tf.get_variable('weights', shape=[3, 3, 16, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[16], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # 第二层池化
    with tf.variable_scope('pooling2_lrn') as scope:
        # 先规范化再池化
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # 第三层全连接 连接所有的特征, 将输出值给分类器, 该层映射出256个输出
    with tf.variable_scope('local3') as scope:
        # 将pool2张量铺平, 再把维度调整成shape(shape里的-1, 程序运行时会自动计算填充)
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        # 获取reshape后的列数
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 256], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[256], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 激活函数为relu
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='local3')

    # 第四层为全连接 该层映射出512个输出
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[256, 512], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[512], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # 第五层为输出
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[512, n_classes], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    return softmax_linear


def losses(logits, labels):
    """
    计算损失函数
    :param logits: 网络输出值
    :param labels: 图片对应标签
    :return: 损失函数值
    """
    with tf.variable_scope('loss') as scope:
        # label与神经网络输出层的输出结果做对比，得到损失值
        # 归一化和交叉熵处理
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求得batch的平均loss
    return loss


def training(loss, learning_rate):
    """
    反向传播
    :param loss: 损失函数值
    :param learning_rate: 学习率
    :return: 训练最优值
    """
    with tf.name_scope('optimizer'):
        # 利用指数衰减法进行参数调整
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 梯度下降一次加1，一般用于记录迭代优化的次数，主要用于参数输出和保存
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # loss：即最小化的目标变量，一般就是训练的目标函数，均方差或者交叉熵
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    计算网络准确率
    :param logits:网络输出值
    :param labels: 真实值
    :return:
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy
