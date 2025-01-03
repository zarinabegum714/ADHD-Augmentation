import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Generator model (U-Net architecture)
def build_generator():
    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d
    def deconv2d(layer_input, skip_input, filters, f_size=4):
        """Layers used during upsampling"""
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        u = Activation('relu')(u)
        u = Concatenate()([u, skip_input])
        return u
    # Image input
    d0 = Input(shape=(128, 128, 3))
    # Downsampling
    d1 = conv2d(d0, 64)
    d2 = conv2d(d1, 128)
    d3 = conv2d(d2, 256)
    d4 = conv2d(d3, 512)
    # Upsampling
    u1 = deconv2d(d4, d3, 256)
    u2 = deconv2d(u1, d2, 128)
    u3 = deconv2d(u2, d1, 64)
    u4 = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same')(u3)
    output_img = Activation('tanh')(u4)
    return Model(d0, output_img)
# Discriminator model
def build_discriminator():
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = tf.keras.layers.InstanceNormalization()(d)
        return d
    img = Input(shape=(128, 128, 3))
    d1 = d_layer(img, 64, normalization=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    d4 = d_layer(d3, 512)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    return Model(img, validity)
# Build and compile the discriminators
d_A = build_discriminator()
d_B = build_discriminator()
d_A.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
d_B.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
# Build the generators
g_AB = build_generator()
g_BA = build_generator()
# Input images from both domains
img_A = Input(shape=(128, 128, 3))
img_B = Input(shape=(128, 128, 3))
# Translate images to the other domain
fake_B = g_AB(img_A)
fake_A = g_BA(img_B)
# Translate images back to original domain
reconstr_A = g_BA(fake_B)
reconstr_B = g_AB(fake_A)
# Identity mapping
img_A_id = g_BA(img_A)
img_B_id = g_AB(img_B)
# For the combined model we will only train the generators
d_A.trainable = False
d_B.trainable = False
# Discriminators determines validity of translated images
valid_A = d_A(fake_A)
valid_B = d_B(fake_B)
# Combined model
combined = Model(inputs=[img_A, img_B],
                 outputs=[valid_A, valid_B,
                          reconstr_A, reconstr_B,
                          img_A_id, img_B_id])
combined.compile(loss=['mse', 'mse',
                       'mae', 'mae',
                       'mae', 'mae'],
                 loss_weights=[1, 1,
                               10, 10,
                               1, 1],
                 optimizer=Adam(0.0002, 0.5))
# Training the CycleGAN
def train(epochs, batch_size=1, save_interval=50):
    # Load the dataset
    (x_trainA, _), (x_trainB, _) = tf.keras.datasets.cifar10.load_data()
    x_trainA = x_trainA.astype('float32') / 255.
    x_trainB = x_trainB.astype('float32') / 255.
    # Rescale -1 to 1
    x_trainA = (x_trainA - 0.5) * 2
    x_trainB = (x_trainB - 0.5) * 2
    # Training loop
    for epoch in range(epochs):
        for batch_i in range(len(x_trainA) // batch_size):
            # Select a random half batch of images
            idx = np.random.randint(0, x_trainA.shape[0], batch_size)
            imgs_A = x_trainA[idx]
            imgs_B = x_trainB[idx]
            # Translate images to opposite domain
            fake_B = g_AB.predict(imgs_A)
            fake_A = g_BA.predict(imgs_B)
            # Train the discriminators
            dA_loss_real = d_A.train_on_batch(imgs_A, np.ones((batch_size, 16, 16, 1)))
            dA_loss_fake = d_A.train_on_batch(fake_A, np.zeros((batch_size, 16, 16, 1)))
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
            dB_loss_real = d_B.train_on_batch(imgs_B, np.ones((batch_size, 16, 16, 1)))
            dB_loss_fake = d_B.train_on_batch(fake_B, np.zeros((batch_size, 16, 16, 1)))
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
            # Total discriminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)
            # Train the generators
            g_loss = combined.train_on_batch([imgs_A, imgs_B],
                                             [np.ones((batch_size, 16, 16, 1)), np.ones((batch_size, 16, 16, 1)),
                                              imgs_A, imgs_B,
                                              imgs_A, imgs_B])
            # Print the progress
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}] [G loss: {g_loss[0]:.4f}]")
            # If at save interval => save generated image samples
            if batch_i % save_interval == 0:
                save_imgs(epoch, batch_i)
# Function to save generated images
def save_imgs(epoch, batch_i):
    r, c = 2, 3
    imgs_A = x_trainA[np.random.randint(0, x_trainA.shape[0], 1)]
    imgs_B = x_trainB[np.random.randint(0, x_trainB.shape[0], 1)]
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)
    reconstr_A = g_BA.predict(fake_B)
    reconstr_B = g_AB.predict(fake_A)
    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"images/{epoch}_{batch_i}.png")
    plt.close()
# Train the CycleGAN
train(epochs=200, batch_size=1, save_interval=200)
