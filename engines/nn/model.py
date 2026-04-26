import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)

    return x


def build_model(input_shape=(8, 8, 18), filters=64, num_blocks=3):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(num_blocks):
        x = residual_block(x, filters)

    # Head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1, activation='tanh')(x)  # выход в [-1, 1]

    model = models.Model(inputs=inputs, outputs=x)

    return model



model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)
#model.summary()



from batch_generator import batch_generator
dataset = tf.data.Dataset.from_generator(
    batch_generator,
    output_signature=(
        tf.TensorSpec(shape=(256, 8, 8, 18), dtype=tf.float32),
        tf.TensorSpec(shape=(256,), dtype=tf.float32),
    )
).take(100).prefetch(tf.data.AUTOTUNE)



model.fit(
    dataset,
    epochs=10
)