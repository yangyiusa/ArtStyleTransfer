import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")

# Enable mixed precision for performance
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Constants
batch_size = 1
epochs = 50
stylized_images_dir = "stylized_images_extended"

# Paths to dataset directories
monet_dir = os.path.expanduser("~/Downloads/Project/dataset/style_images")
photo_dir = os.path.expanduser("~/Downloads/Project/dataset/content_images")

# Preprocessing function
def preprocess_image_tf(image_path, target_size=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return image

# Create datasets for Monet and photo images
monet_ds = tf.data.Dataset.list_files(os.path.join(monet_dir, "*.jpg"))
monet_ds = monet_ds.map(lambda x: preprocess_image_tf(x), num_parallel_calls=tf.data.AUTOTUNE).cache()

photo_ds = tf.data.Dataset.list_files(os.path.join(photo_dir, "*.jpg"))
photo_ds = photo_ds.map(lambda x: preprocess_image_tf(x), num_parallel_calls=tf.data.AUTOTUNE).cache()

# Data Augmentation
monet_ds = monet_ds.map(lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=tf.data.AUTOTUNE)
photo_ds = photo_ds.map(lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch for performance
monet_ds = monet_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
photo_ds = photo_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


print("Datasets are ready.")

# Residual block for the generator
def residual_block(x):
    res = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    res = layers.Conv2D(256, kernel_size=3, padding="same")(res)
    return layers.add([x, res])

# Generator with residual blocks
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same", activation="relu")(x)

    for _ in range(9):
        x = residual_block(x)

    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(3, kernel_size=7, padding="same", activation="tanh")(x)

    return Model(inputs, outputs)

# Discriminator
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, kernel_size=4, padding="same")(x)

    return Model(inputs, outputs)

# Instantiate the models
G = build_generator()  # Photo → Monet
F = build_generator()  # Monet → Photo
D_X = build_discriminator()  # Discriminator for Monet
D_Y = build_discriminator()  # Discriminator for Photo

print("CycleGAN models created.")

# Loss functions
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def adversarial_loss(real, generated):
    return mse(real, generated)

def cycle_consistency_loss(real, cycled):
    return mae(real, cycled)

# Learning Rate Scheduler for Generator and Discriminator
generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0002,
    decay_steps=10000,
    decay_rate=0.9
)
discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0002,
    decay_steps=10000,
    decay_rate=0.9
)

# Updated Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)


# Train Step
@tf.function
def train_step(real_photo, real_monet):
    real_target = 0.9  # Smooth the label for real images
    fake_target = 0.0  # Fake labels stay the same

    with tf.GradientTape(persistent=True) as tape:
        # Generate fake images and cycle them
        fake_monet = G(real_photo, training=True)
        cycled_photo = F(fake_monet, training=True)

        fake_photo = F(real_monet, training=True)
        cycled_monet = G(fake_photo, training=True)

        # Generator losses
        gen_g_loss = adversarial_loss(D_X(fake_monet, training=True), tf.ones_like(D_X(fake_monet)) * real_target)
        gen_f_loss = adversarial_loss(D_Y(fake_photo, training=True), tf.ones_like(D_Y(fake_photo)) * real_target)
        cycle_loss = cycle_consistency_loss(real_photo, cycled_photo) + cycle_consistency_loss(real_monet, cycled_monet)
        lambda_cycle = 10.0
        total_gen_loss = gen_g_loss + gen_f_loss + lambda_cycle * cycle_loss

        # Discriminator losses
        real_loss_x = adversarial_loss(D_X(real_monet, training=True), tf.ones_like(D_X(real_monet)) * real_target)
        fake_loss_x = adversarial_loss(D_X(fake_monet, training=True), tf.zeros_like(D_X(fake_monet)) + fake_target)
        real_loss_y = adversarial_loss(D_Y(real_photo, training=True), tf.ones_like(D_Y(real_photo)) * real_target)
        fake_loss_y = adversarial_loss(D_Y(fake_photo, training=True), tf.zeros_like(D_Y(fake_photo)) + fake_target)
        total_disc_loss = real_loss_x + fake_loss_x + real_loss_y + fake_loss_y

    # Clip and apply gradients
    generator_gradients = tape.gradient(total_gen_loss, G.trainable_variables + F.trainable_variables)
    generator_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in generator_gradients]
    discriminator_gradients = tape.gradient(total_disc_loss, D_X.trainable_variables + D_Y.trainable_variables)
    discriminator_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in discriminator_gradients]

    generator_optimizer.apply_gradients(zip(generator_gradients, G.trainable_variables + F.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D_X.trainable_variables + D_Y.trainable_variables))

# Training Loop
steps = min(tf.data.experimental.cardinality(photo_ds).numpy(), tf.data.experimental.cardinality(monet_ds).numpy())
for epoch in range(epochs):
    for step, (photo, monet) in enumerate(tf.data.Dataset.zip((photo_ds, monet_ds)).take(steps)):
        train_step(photo, monet)
        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}/{steps} completed.")

    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Generate and Save Stylized Images
output_dir = "stylized_images_extended"
os.makedirs(output_dir, exist_ok=True)

photo_count = tf.data.experimental.cardinality(photo_ds).numpy()
for i, photo_batch in enumerate(photo_ds):  # Iterate through the dataset

    # Ensure photo_batch has the correct shape (add batch dimension if missing)
    if len(photo_batch.shape) == 3:  # If shape is (256, 256, 3)
        photo_batch = tf.expand_dims(photo_batch, axis=0)  # Add batch dimension


    # Process the batch through the generator
    stylized_images = G(photo_batch, training=False).numpy()
    stylized_images = ((stylized_images + 1.0) * 127.5).astype("uint8")

    # Save each image in the batch
    for j, img in enumerate(stylized_images):
        output_path = os.path.join(output_dir, f"image_{i * batch_size + j + 1}.jpg")
        tf.keras.utils.save_img(output_path, img)

print(f"Generated and saved {photo_count} stylized images in {output_dir}.")

# Visualize results
num_samples = 10  # Increase to visualize more samples
sample_photos = list(photo_ds.unbatch().take(num_samples))  # Convert to list

fig, axs = plt.subplots(len(sample_photos), 2, figsize=(10, len(sample_photos) * 4))

for i, photo in enumerate(sample_photos):
    photo_batch = tf.expand_dims(photo, axis=0)  # Add batch dimension explicitly
    stylized_image = G(photo_batch, training=False).numpy()[0]  # Remove batch dimension after processing

    # Denormalize and convert to uint8
    photo_image = ((photo.numpy() + 1.0) * 127.5).astype("uint8")
    stylized_image = ((stylized_image + 1.0) * 127.5).astype("uint8")

    # Plot original and stylized images
    axs[i, 0].imshow(photo_image)
    axs[i, 0].axis("off")
    axs[i, 0].set_title("Input Photo")

    axs[i, 1].imshow(stylized_image)
    axs[i, 1].axis("off")
    axs[i, 1].set_title("Stylized Image")

plt.tight_layout()
plt.savefig("stylized_image_visualization_more.png")
plt.show()

G.save_weights("generator_weights.h5")
