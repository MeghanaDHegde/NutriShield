import streamlit as st
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import json
import requests
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Function to load the pre-trained EfficientNetB0 model
def load_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    num_classes = 79  # Replace with the actual number of classes you have
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load your trained model (Ensure you have the model file saved as 'food_model.keras')
model = load_model()


# Function to preprocess the image before feeding it to the model
def preprocess_image(img):
    # Resize and preprocess the image as expected by the EfficientNet model
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)
    return img_array


# Function to fetch food data (using OpenFoodFacts API for nutritional and allergen info)
def fetch_food_data(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Failed to fetch data from Open Food Facts API."}

    data = response.json()

    if "product" in data and data["product"]:
        product = data["product"]
        nutrition = product.get("nutriments", {})
        allergens = product.get("allergens", "No allergens listed.")
        ingredients = product.get("ingredients_text", "No ingredients found.")

        nutritional_info = {
            "energy_kcal": nutrition.get("energy-kcal", "Not available"),
            "fat": nutrition.get("fat", "Not available"),
            "saturated_fat": nutrition.get("saturated-fat", "Not available"),
            "carbohydrates": nutrition.get("carbohydrates", "Not available"),
            "sugars": nutrition.get("sugars", "Not available"),
            "fiber": nutrition.get("fiber", "Not available"),
            "proteins": nutrition.get("proteins", "Not available"),
            "salt": nutrition.get("salt", "Not available"),
        }

        return {
            "nutritional_info": nutritional_info,
            "allergens": allergens,
            "ingredients": ingredients
        }

    return {"error": "Product not found or missing data."}


# Detect allergens in ingredients
def detect_allergens_in_ingredients(ingredients):
    allergens_keywords = ["peanut", "milk", "gluten", "egg", "soy", "wheat", "fish", "shellfish", "nuts", "sesame"]
    allergens_found = [allergen.capitalize() for allergen in allergens_keywords if allergen in ingredients.lower()]
    return allergens_found


# Streamlit app code for "Food Image Recognition" screen
def app():
    st.title("Food Image Recognition and Barcode Scanner")

    # Set app_mode to 'Food Image Recognition' for testing purposes
    app_mode = st.sidebar.selectbox("Select Mode", ["Food Image Recognition", "Barcode Scanner"])

    if app_mode == "Food Image Recognition":
        st.header("Food Image Recognition")
        st.write("Upload an image or use your camera to classify food.")

        # Option to use webcam or upload an image
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        # Capture image using camera
        img_file = st.camera_input("Take a picture")

        # Handle both uploaded image and camera input
        img = None
        if uploaded_img:
            # If user uploaded an image, convert it to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, caption="Uploaded Image", use_column_width=True)

        elif img_file:
            # If user used the camera, convert it to OpenCV format
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, caption="Captured Image", use_column_width=True)

        if img is not None:
            # Preprocess the image before feeding it to the model
            preprocessed_img = preprocess_image(img)

            # Make a prediction using the model
            predictions = model.predict(preprocessed_img)
            predicted_class = np.argmax(predictions, axis=1)  # Get the predicted class

            # Define your class labels (change this according to your trained model's output classes)
            class_labels = ['Apple', 'Banana', 'Carrot', 'Tomato']  # Replace with actual labels
            predicted_food = class_labels[predicted_class[0]]

            st.write(f"Predicted food item: **{predicted_food}**")

            # Fetch additional nutritional or allergen information
            food_data = fetch_food_data(predicted_food)
            if "error" in food_data:
                st.error(food_data["error"])
            else:
                st.subheader("🍽️ Nutritional Information:")
                st.write(f"🔥 **Energy (kcal)**: {food_data['nutritional_info']['energy_kcal']}")
                st.write(f"🥚 **Fat**: {food_data['nutritional_info']['fat']} g")
                st.write(f"🥚 **Saturated Fat**: {food_data['nutritional_info']['saturated_fat']} g")
                st.write(f"🍞 **Carbohydrates**: {food_data['nutritional_info']['carbohydrates']} g")
                st.write(f"🍬 **Sugars**: {food_data['nutritional_info']['sugars']} g")
                st.write(f"🍠 **Fiber**: {food_data['nutritional_info']['fiber']} g")
                st.write(f"🍗 **Proteins**: {food_data['nutritional_info']['proteins']} g")
                st.write(f"🧂 **Salt**: {food_data['nutritional_info']['salt']} g")
                st.markdown("---")

                # Allergens check
                allergens_from_ingredients = detect_allergens_in_ingredients(food_data["ingredients"])
                st.subheader("⚠️ Allergens:")
                if allergens_from_ingredients:
                    allergens_str = ", ".join(allergens_from_ingredients)
                    st.warning(f"⚠️ **Allergen Warning**: Contains allergens: {allergens_str}")

                # Health Risk Alert based on user's allergies
                if 'allergies' in st.session_state.user_profile:
                    user_allergies = set(st.session_state.user_profile['allergies'])
                    risk_allergens = set(allergens_from_ingredients) & user_allergies
                    if risk_allergens:
                        st.warning(
                            f"⚠️ **Risk Alert**: Your medical record indicates allergies to {', '.join(risk_allergens)}. Please proceed with caution!")

                # Display Ingredients Information
                st.subheader("🍽️ Ingredients Information:")
                st.write(food_data["ingredients"])


# Run the Streamlit app
if __name__ == "__main__":
    app()
