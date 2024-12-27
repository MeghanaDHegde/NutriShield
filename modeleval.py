import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Load EfficientNetB0 model pre-trained on ImageNet (excluding the top layers)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model to avoid updating pre-trained weights during initial training
base_model.trainable = False

# Step 2: Add custom classification layers on top of EfficientNetB0
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling to flatten features
x = Dense(1024, activation='relu')(x)  # Dense layer with 1024 units
x = Dense(256, activation='relu')(x)  # Dense layer with 256 units
num_classes =  79 # Change this to the number of food classes in your dataset (e.g., 101 for Food-101)

# Output layer with softmax activation for multi-class classification
predictions = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Step 3: Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Set up data augmentation and preprocessing for your dataset
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,  # Preprocessing function for EfficientNet
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

# Set up ImageDataGenerators for training and validation data
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\Meghana D Hegde\Downloads\archive (6)\dataset\train",  # Replace with your training data directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Multi-class classification
)

valid_generator = valid_datagen.flow_from_directory(
    r"C:\Users\Meghana D Hegde\Downloads\archive (6)\dataset\test",  # Replace with your validation data directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Multi-class classification
)

# Step 5: Set up callbacks for early stopping and model checkpoints
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('food_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')


# Step 6: Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs based on your dataset size
    validation_data=valid_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[early_stopping, checkpoint]
)

# Step 7: Fine-tune the model (Optional)
# Unfreeze the last few layers of EfficientNetB0 for fine-tuning
base_model.trainable = True

# Optionally, unfreeze only the last few layers of EfficientNetB0 (e.g., unfreeze the last 4 blocks)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile the model after unfreezing layers
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model with the unfrozen layers
history_finetune = model.fit(
    train_generator,
    epochs=5,  # Continue training for more epochs
    validation_data=valid_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

# Step 8: Evaluate the fine-tuned model
loss, accuracy = model.evaluate(valid_generator)
print(f"Fine-tuned model accuracy: {accuracy * 100:.2f}%")
