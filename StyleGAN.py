import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define the resolution of the images
resolution = 64
latent_dim = 100
# Mapping Network
def build_mapping_network():
    model = tf.keras.Sequential()
    model.add(Dense(512, input_dim=latent_dim, activation='relu'))
    for _ in range(7):
        model.add(Dense(512, activation='relu'))
    return model
# Synthesis Network
def build_synthesis_network():
    model = tf.keras.Sequential()
    model.add(Dense(4 * 4 * 512, input_dim=512))
    model.add(Reshape((4, 4, 512)))
    for _ in range(4):
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(512, kernel_size=3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))
    return model
# Discriminator
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(Conv2D(64, kernel_size=3, padding='same', input_shape=(resolution, resolution, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    for _ in range(4):
        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model
# Build the models
mapping_network = build_mapping_network()
synthesis_network = build_synthesis_network()
discriminator = build_discriminator()
# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
# Input for latent vector
z = Input(shape=(latent_dim,))
w = mapping_network(z)
img = synthesis_network(w)
# For the combined model, only train the generator
discriminator.trainable = False
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
# Training the StyleGAN
def train(epochs, batch_size=32, save_interval=50):
    # Load the dataset (using CIFAR-10 for simplicity)
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 127.5 - 1.0  # Normalize to [-1, 1]

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random half batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]
        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = synthesis_network.predict(mapping_network.predict(noise))
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
            pass
            # save_imgs(epoch)
# Function to save generated images
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = synthesis_network.predict(mapping_network.predict(noise))

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    # fig.savefig(f"images/stylegan_{epoch}.png")
    plt.close()

# Train the StyleGAN
train(epochs=100, batch_size=32, save_interval=200)
