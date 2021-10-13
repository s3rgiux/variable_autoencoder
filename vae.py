from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder, vae = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    _, _, z = encoder.predict(x_test,
                                   batch_size=batch_size)
    images = decoder.predict(z)
    images = vae.predict(x_test)


    for i in range(images.shape[0]):
        x = (np.squeeze(x_test[i]*255)).astype("uint8")
        y_true = (np.squeeze(y_test[i]*255)).astype("uint8")
        y_pred = (np.squeeze(images[i]*255)).astype("uint8")
        
        print(x.max(), y_pred.max(), y_true.max())

        cv2.imwrite(f"./output/{i}_x.png", x)
        print("written")
        cv2.imwrite(f"./output/{i}_y.png", y_true)
        print("writtten2")
        cv2.imwrite(f"./output/{i}_yhat.png", y_pred)

    # plt.figure(figsize=(12, 10))
    # plt.scatter(z_mean[:, 0], z_mean[:, 1])
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    # filename = os.path.join(model_name, "digits_over_latent.png")
    # # display a 30x30 2D manifold of digits
    # n = 30
    # digit_size = 28
    # figure = np.zeros((480 * n, 320 * n))
    # # linearly spaced coordinates corresponding to the 2D plot
    # # of digit classes in the latent space
    # grid_x = np.linspace(-4, 4, n)
    # grid_y = np.linspace(-4, 4, n)[::-1]

    # for i, yi in enumerate(grid_y):
    #     for j, xi in enumerate(grid_x):
    #         z_sample = np.array([[xi, yi]])
    #         x_decoded = decoder.predict(z_sample)
    #         digit = x_decoded[0].reshape(480, 320)
    #         figure[i * 480: (i + 1) * 480,
    #                j * 320: (j + 1) * 320] = digit

    # plt.figure(figsize=(10, 10))
    # start_range = 480 // 2
    # end_range = n * 480 + start_range + 1
    # pixel_x_range = np.arange(start_range, end_range, 480)
    # start_range = 320 // 2
    # end_range = n * 320 + start_range + 1
    # pixel_y_range = np.arange(start_range, end_range, 320)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_y_range, sample_range_x)
    # plt.yticks(pixel_x_range, sample_range_y)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.imshow(figure, cmap='Greys_r')
    # plt.savefig(filename)
    # plt.show()


# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, y_train, x_test, y_test

# image_size = x_train.shape[1]
# x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
# x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
def generator(folder, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 480, 320))
    batch_labels = np.zeros((batch_size, 480, 320))

    images_map_list = list(glob.glob(f'{folder}/*_map.png'))
    images_depth_list = list(glob.glob(f'{folder}/*_img.png'))

    assert len(images_depth_list) == len(images_map_list), "different lenght"
    print(len(images_depth_list))

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(images_map_list),1)[0]
            
            file_map = images_map_list[index]
            file_img = images_depth_list[index]

            map_image = cv2.imread(file_map, 0)
            depth_image = cv2.imread(file_img, 0)

            output = np.vstack([map_image/255, depth_image/255])


            row,col = map_image.shape
            gauss = np.random.randn(row,col)*.1
            gauss = gauss.reshape(row,col)        
            # map_image = map_image + map_image * gauss
            map_image = np.clip(map_image/255 + gauss,0,1)
            input = np.vstack([map_image, depth_image/255])
            
            batch_features[i] = input
            batch_labels[i] = output
            
            # cv2.imshow("in", input)
            # cv2.imshow("out", output)
            # cv2.waitKey(1)

        img_h, img_w = input.shape
        x = np.reshape(batch_features, [-1, img_h, img_w, 1])
        y = np.reshape(batch_labels, [-1, img_h, img_w, 1])
        yield x, y


def test_data(folder, batch_size):
     # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 480, 320))
    batch_labels = np.zeros((batch_size, 480, 320))

    images_map_list = list(glob.glob(f'{folder}/*_map.png'))
    images_depth_list = list(glob.glob(f'{folder}/*_img.png'))

    assert len(images_depth_list) == len(images_map_list), "different lenght"
    print(len(images_depth_list))

    for i in range(batch_size):
        # choose random index in features
        index = np.random.choice(len(images_map_list),1)[0]
        
        file_map = images_map_list[index]
        file_img = images_depth_list[index]

        map_image = cv2.imread(file_map, 0)
        depth_image = cv2.imread(file_img, 0)

        output = np.vstack([map_image/255, depth_image/255])


        row,col = map_image.shape
        gauss = np.random.randn(row,col)*.1
        gauss = gauss.reshape(row,col)        
        # map_image = map_image + map_image * gauss
        map_image = np.clip(map_image/255 + gauss,0,1)
        input = np.vstack([map_image, depth_image/255])
        
        batch_features[i] = input
        batch_labels[i] = output
        
        #cv2.imshow("in", input)
        #cv2.imshow("out", output)
        #cv2.waitKey(1)

    img_h, img_w = input.shape
    x = np.reshape(batch_features, [-1, img_h, img_w, 1])
    y = np.reshape(batch_labels, [-1, img_h, img_w, 1])
    return x, y

# network parameters
input_shape = (480, 320, 1)
batch_size = 8#16
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 1

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# classifier
# c = Input(shape=(latent_dim,), name='z_class_sampling')
# c1 = Dense(shape[1] * shape[2] * shape[3], activation='relu')(c)
# c2 = Dense(shape[1] * shape[2], activation='relu')(c1)
# c3 = Dense(shape[1], activation='relu')(c2)
# c4 = Dense(3, activation='softmax')(c3)
# classifier = Model(c, c4, name="classifier")


# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# out_class = classifier(encoder(inputs)[2])
# vae_classifier = Model(inputs, out_class, name='vae_classifier')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder, vae)
    #data = test_data("/media/infinity0106/UBUNTU 18_0/trainning", batch_size)
    data = test_data("/home/sergio/trainning", batch_size)

    # # VAE loss = mse_loss or xent_loss + kl_loss
    # reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))

    # reconstruction_loss *= 320 * 480
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)
    # vae.add_loss(vae_loss)
    # vae.compile(optimizer='rmsprop')
    def my_vae_loss(y_true, y_pred):
        xent_loss = 320 * 480 * mse(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss

    vae.compile(optimizer='rmsprop', loss=my_vae_loss)
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        print("entre")
        # train the autoencoder
        vae.load_weights('vae_cnn_mnist.h5')
        vae.fit_generator(generator("/home/sergio/trainning", batch_size),
                steps_per_epoch=3400//batch_size,
                # steps_per_epoch=1,
                epochs=epochs)
        vae.save_weights('vae_cnn_mnist.h5')
    print("finish")
    plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")
    print("finish2")