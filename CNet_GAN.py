import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, multiply, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Generator model
def build_generator(latent_dim, num_classes):
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    noise = Input(shape=(latent_dim,))
    model_input = multiply([noise, label_embedding])
    x = Dense(4 * 4 * 512)(model_input)
    x = Reshape((4, 4, 512))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    img = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
    return Model([noise, label], img)
# Discriminator model
def build_discriminator(img_shape, num_classes):
    img = Input(shape=img_shape)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(img)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)
    model_input = multiply([flat_img, label_embedding])
    x = Dense(512)(model_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    validity = Dense(1, activation='sigmoid')(x)
    return Model([img, label], validity)
# Training the CNet-GAN
def train(epochs, batch_size=32, save_interval=200, latent_dim=100):
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 127.5 - 1.0  # Normalize to [-1, 1]
    y_train = y_train.flatten()
    num_classes = 10
    img_shape = x_train.shape[1:]
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    # Build and compile the discriminator
    discriminator = build_discriminator(img_shape, num_classes)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    # Build the generator
    generator = build_generator(latent_dim, num_classes)
    # The generator takes noise and the target label as input
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    img = generator([noise, label])
    # For the combined model, only train the generator
    discriminator.trainable = False
    # The discriminator takes generated image and label as input
    valid = discriminator([img, label])
    combined = Model([noise, label], valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    # Training loop
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random half batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs, labels = x_train[idx], y_train[idx]
        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_labels = np.random.randint(0, num_classes, batch_size)
        gen_imgs = generator.predict([noise, gen_labels])
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, gen_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_labels = np.random.randint(0, num_classes, batch_size)
        # Train the generator (to have the discriminator label samples as valid)
        g_loss = combined.train_on_batch([noise, gen_labels], valid)
        # Print the progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            pass
            # save_imgs(epoch, generator, latent_dim, num_classes)
# Function to save generated images
def save_imgs(epoch, generator, latent_dim, num_classes):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_labels = np.random.randint(0, num_classes, r * c)
    gen_imgs = generator.predict([noise, gen_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(f"Label: {gen_labels[cnt]}")
            axs[i, j].axis('off')
            cnt += 1
    # fig.savefig(f"images/cnetgan_{epoch}.png")
    plt.close()
# Train the CNet-GAN
train(epochs=100, batch_size=32, save_interval=200)
