import tensorflow as tf
from tensorflow.keras import layers, models

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)

###############################################  mlp #############################################
class G_mlp(tf.keras.Model):
    def __init__(self):
        super(G_mlp, self).__init__()
        self.name_scope = 'G_mlp'
        
        self.dense1 = layers.Dense(4 * 4 * 512, activation=lrelu)
        self.bn1 = layers.BatchNormalization()
        
        self.dense2 = layers.Dense(64, activation=lrelu)
        self.bn2 = layers.BatchNormalization()
        
        self.dense3 = layers.Dense(64, activation=lrelu)
        self.bn3 = layers.BatchNormalization()
        
        self.dense4 = layers.Dense(64 * 64 * 3, activation=tf.nn.tanh)
        self.bn4 = layers.BatchNormalization()
        
        self.reshape = layers.Reshape((64, 64, 3))

    def call(self, z, training=True):
        x = self.dense1(z)
        x = self.bn1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        
        x = self.dense4(x)
        x = self.bn4(x, training=training)
        
        return self.reshape(x)

class D_mlp(tf.keras.Model):
    def __init__(self):
        super(D_mlp, self).__init__()
        self.name_scope = "D_mlp"
        
        self.flatten = layers.Flatten()
        
        self.dense1 = layers.Dense(64, activation=tf.nn.relu)
        self.bn1 = layers.BatchNormalization()
        
        self.dense2 = layers.Dense(64, activation=tf.nn.relu)
        self.bn2 = layers.BatchNormalization()
        
        self.dense3 = layers.Dense(64, activation=tf.nn.relu)
        self.bn3 = layers.BatchNormalization()
        
        self.dense4 = layers.Dense(1)  # Output logit

    def call(self, x, training=True):
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        
        logit = self.dense4(x)
        
        return logit

#-------------------------------- MNIST for test ------
class G_mlp_mnist(tf.keras.Model):
    def __init__(self):
        # super(G_mlp_mnist, self).__init__()
        # self.name_scope = "G_mlp_mnist"
        super(G_mlp_mnist, self).__init__(name="G_mlp_mnist")  # 使用 name 参数命名模型
        self.X_dim = 784
        
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.dense1 = layers.Dense(128, activation=tf.nn.relu, kernel_initializer=initializer)
        self.dense2 = layers.Dense(self.X_dim, activation=tf.nn.sigmoid, kernel_initializer=initializer)

    def call(self, z, training=True):
        g = self.dense1(z)
        g = self.dense2(g)
        return g

class D_mlp_mnist(tf.keras.Model):
    def __init__(self):
        super(D_mlp_mnist, self).__init__(name="D_mlp_mnist")
        # self.name_scope = "D_mlp_mnist"
        
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.shared = layers.Dense(128, activation=tf.nn.relu, kernel_initializer=initializer)
        self.d = layers.Dense(1, kernel_initializer=initializer)  # Output logit
        self.q = layers.Dense(10, kernel_initializer=initializer)  # 10 classes

    def call(self, x, training=True):
        shared = self.shared(x)
        d = self.d(shared)
        q = self.q(shared)
        return d, q

class Q_mlp_mnist(tf.keras.Model):
    def __init__(self):
        super(Q_mlp_mnist, self).__init__(name="Q_mlp_mnist")
        # self.name_scope = "Q_mlp_mnist"
        
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.shared = layers.Dense(128, activation=tf.nn.relu, kernel_initializer=initializer)
        self.q = layers.Dense(10, kernel_initializer=initializer)  # 10 classes

    def call(self, x, training=True):
        shared = self.shared(x)
        q = self.q(shared)
        return q

###############################################  conv #############################################
class G_conv(tf.keras.Model):
    def __init__(self):
        super(G_conv, self).__init__()
        self.name_scope = 'G_conv'
        self.size = 64 // 16
        self.channel = 3
        
        self.dense = layers.Dense(self.size * self.size * 1024, activation=tf.nn.relu)
        self.bn0 = layers.BatchNormalization()
        self.reshape = layers.Reshape((self.size, self.size, 1024))
        
        self.conv_transpose1 = layers.Conv2DTranspose(512, 3, strides=2, padding='SAME', 
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))
        self.bn1 = layers.BatchNormalization()
        
        self.conv_transpose2 = layers.Conv2DTranspose(256, 3, strides=2, padding='SAME', 
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))
        self.bn2 = layers.BatchNormalization()
        
        self.conv_transpose3 = layers.Conv2DTranspose(128, 3, strides=2, padding='SAME', 
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))
        self.bn3 = layers.BatchNormalization()
        
        self.conv_transpose4 = layers.Conv2DTranspose(self.channel, 3, strides=2, padding='SAME', 
                                                     activation=tf.nn.sigmoid,
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))

    def call(self, z, training=True):
        g = self.dense(z)
        g = self.bn0(g, training=training)
        g = self.reshape(g)
        
        g = self.conv_transpose1(g)
        g = self.bn1(g, training=training)
        
        g = self.conv_transpose2(g)
        g = self.bn2(g, training=training)
        
        g = self.conv_transpose3(g)
        g = self.bn3(g, training=training)
        
        g = self.conv_transpose4(g)
        
        return g

class D_conv(tf.keras.Model):
    def __init__(self):
        super(D_conv, self).__init__()
        self.name_scope = 'D_conv'
        
        size = 64
        self.conv1 = layers.Conv2D(size, 4, strides=2, activation=lrelu)
        
        self.conv2 = layers.Conv2D(size * 2, 4, strides=2, activation=lrelu)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(size * 4, 4, strides=2, activation=lrelu)
        self.bn3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(size * 8, 4, strides=2, activation=lrelu)
        self.bn4 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        
        self.d = layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02))
        
        self.q_dense = layers.Dense(128, activation=lrelu)
        self.q_bn = layers.BatchNormalization()
        self.q = layers.Dense(2)  # 2 classes

    def call(self, x, training=True):
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        
        x = self.flatten(x)
        
        d = self.d(x)
        
        q = self.q_dense(x)
        q = self.q_bn(q, training=training)
        q = self.q(q)
        
        return d, q

# -------------------------------- MNIST for test
class G_conv_mnist(tf.keras.Model):
    def __init__(self):
        super(G_conv_mnist, self).__init__()
        self.name_scope = 'G_conv_mnist'
        
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.dense = layers.Dense(7 * 7 * 128, activation=tf.nn.relu, kernel_initializer=initializer)
        self.bn0 = layers.BatchNormalization()
        
        self.reshape = layers.Reshape((7, 7, 128))
        
        self.conv_transpose1 = layers.Conv2DTranspose(64, 4, strides=2, padding='SAME', 
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=initializer)
        self.bn1 = layers.BatchNormalization()
        
        self.conv_transpose2 = layers.Conv2DTranspose(1, 4, strides=2, padding='SAME', 
                                                     activation=tf.nn.sigmoid,
                                                     kernel_initializer=initializer)

    def call(self, z, training=True):
        g = self.dense(z)
        g = self.bn0(g, training=training)
        
        g = self.reshape(g)
        
        g = self.conv_transpose1(g)
        g = self.bn1(g, training=training)
        
        g = self.conv_transpose2(g)
        
        return g

class D_conv_mnist(tf.keras.Model):
    def __init__(self):
        super(D_conv_mnist, self).__init__()
        self.name_scope = 'D_conv_mnist'
        
        size = 64
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.conv1 = layers.Conv2D(size, 4, strides=2, activation=lrelu)
        
        self.conv2 = layers.Conv2D(size * 2, 4, strides=2, activation=lrelu)
        self.bn2 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        
        self.d = layers.Dense(1, kernel_initializer=initializer)
        
        self.q_dense = layers.Dense(128, activation=lrelu)
        self.q_bn = layers.BatchNormalization()
        self.q = layers.Dense(10)  # 10 classes

    def call(self, x, training=True):
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.flatten(x)
        
        d = self.d(x)
        
        q = self.q_dense(x)
        q = self.q_bn(q, training=training)
        q = self.q(q)
        
        return d, q

class C_conv_mnist(tf.keras.Model):
    def __init__(self):
        super(C_conv_mnist, self).__init__()
        self.name_scope = 'C_conv_mnist'
        
        size = 64
        self.conv1 = layers.Conv2D(size, 5, strides=2, activation=tf.nn.relu)
        
        self.conv2 = layers.Conv2D(size * 2, 5, strides=2, activation=lrelu)
        self.bn2 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        
        self.dense = layers.Dense(1024, activation=tf.nn.relu)
        
        self.c = layers.Dense(10)  # 10 classes

    def call(self, x, training=True):
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.flatten(x)
        
        x = self.dense(x)
        
        c = self.c(x)
        
        return c