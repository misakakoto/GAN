import os
import sys
from PIL import Image
import imageio
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import tensorflow as tf

prefix = './Datas/'

def get_img(img_path, crop_h, resize_h):
    img = imageio.imread(img_path).astype(np.float)
    # crop resize
    crop_w = crop_h
    resize_w = resize_h
    h, w = img.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    cropped_image = np.array(Image.fromarray(img[j:j+crop_h, i:i+crop_w].astype(np.uint8)).resize((resize_h, resize_w)))
    return cropped_image / 255.0

class face3D():
    def __init__(self):
        datapath = '/ssd/fengyao/pose/pose/images'
        self.z_dim = 100
        self.c_dim = 2
        self.size = 64
        self.channel = 3
        self.data = glob.glob(os.path.join(datapath, '*.jpg'))
        self.batch_count = 0

    def __call__(self, batch_size):
        batch_number = len(self.data) // batch_size  # Update division operator for Python 3
        if self.batch_count < batch_number - 1:
            self.batch_count += 1
        else:
            self.batch_count = 0

        path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        batch = [get_img(img_path, 256, self.size) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)
        
        return batch_imgs

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)
        return fig

class celebA():
    def __init__(self):
        datapath = prefix + 'celebA'
        self.z_dim = 100
        self.size = 64
        self.channel = 3
        self.data = glob.glob(os.path.join(datapath, '*.jpg'))
        self.batch_count = 0

    def __call__(self, batch_size):
        batch_number = len(self.data) // batch_size  # Update division operator for Python 3
        if self.batch_count < batch_number - 1:
            self.batch_count += 1
        else:
            self.batch_count = 0

        path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        batch = [get_img(img_path, 128, self.size) for img_path in path_list]
        batch_imgs = np.array(batch).astype(np.float32)
        
        return batch_imgs

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)
        return fig

class mnist():
    def __init__(self, flag='conv', is_tanh=False):
        self.X_dim = 784  # for mlp
        self.z_dim = 100
        self.y_dim = 10
        self.size = 28  # for conv
        self.channel = 1  # for conv
        self.flag = flag
        self.is_tanh = is_tanh
        
        # Load MNIST dataset using tf.keras
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        
        # Normalize images to [0, 1]
        self.images = x_train.astype(np.float32) / 255.0
        
        # Convert labels to one-hot encoding
        self.labels = tf.keras.utils.to_categorical(y_train, num_classes=10)
        
        # Keep track of current position in dataset
        self.current_index = 0
        self.num_examples = len(self.images)

    def __call__(self, batch_size):
        if self.current_index + batch_size > self.num_examples:
            # Shuffle data at the end of an epoch
            indices = np.arange(self.num_examples)
            np.random.shuffle(indices)
            self.images = self.images[indices]
            self.labels = self.labels[indices]
            self.current_index = 0
            
        start = self.current_index
        end = min(start + batch_size, self.num_examples)
        self.current_index = end
        
        batch_imgs = self.images[start:end]
        batch_labels = self.labels[start:end]
        
        if self.flag == 'conv':
            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.size, self.size, self.channel))
        else:
            batch_imgs = np.reshape(batch_imgs, (batch_imgs.shape[0], self.X_dim))
            
        if self.is_tanh:
            batch_imgs = batch_imgs * 2 - 1
            
        return batch_imgs, batch_labels

    def data2fig(self, samples):
        if self.is_tanh:
            samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size, self.size), cmap='Greys_r')
        return fig

if __name__ == '__main__':
    data = face3D()
    print(data(17).shape)