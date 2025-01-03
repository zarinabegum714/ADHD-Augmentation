import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt

# Self-Attention Layer
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query_conv = Conv2D(channels // 8, kernel_size=1, padding='same')
        self.key_conv = Conv2D(channels // 8, kernel_size=1, padding='same')
        self.value_conv = Conv2D(channels, kernel_size=1, padding='same')
        self.gamma = self.add_weight(shape=[], initializer=RandomNormal(0.0, 0.02), trainable=True)
    def call(self, x):
        batch_size, height, width, channels = x.shape
        proj_query = tf.reshape(self.query_conv(x), (batch_size, height * width, -1))
        proj_key = tf.reshape(self.key_conv(x), (batch_size, height * width, -1))
        energy = tf.matmul(proj_query, proj_key, transpose_b=True)
        attention = tf.nn.softmax(energy)
        proj_value = tf.reshape(self.value_conv(x), (batch_size, height * width, channels))
        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, (batch_size, height, width, channels))
        return self.gamma * out + x
# Generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(Dense(4 * 4 * 512, activation="relu", input_dim=latent_dim))
    model.add(Reshape((4, 4, 512)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(SelfAttention(256))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same"))
    model.add(Activation("tanh"))
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)
# Discriminator model
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(SelfAttention(128))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)
# Training the SA-GAN
def train(epochs, batch_size=32, save_interval=50, latent_dim=100):
    # Load the dataset (using CIFAR-10 for simplicity)
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 127.5 - 1.0  # Normalize to [-1, 1]
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    # Build and compile the discriminator
    img_shape = x_train.shape[1:]
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    # Build the generator
    generator = build_generator(latent_dim)
    # The generator takes noise as input and generates imgs
    z = Input(shape=(latent_dim,))
    img = generator(z)
    # For the combined model, only train the generator
    discriminator.trainable = False
    # The discriminator takes generated images as input and determines validity
    valid = discriminator(img)
    # Combined model (stacked generator and discriminator)
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    # Training loop
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random half batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]
        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)
        # Print the progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch, generator, latent_dim)
# Function to save generated images
def save_imgs(epoch, generator, latent_dim):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    # fig.savefig(f"images/sagan_{epoch}.png")
    plt.close()
# Train the SA-GAN
train(epochs=100, batch_size=32, save_interval=200)
