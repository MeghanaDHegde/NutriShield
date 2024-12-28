import streamlit as st
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import os
import streamlit.components.v1 as components
import pandas as pd
import pickle
from PyPDF2 import PdfReader
import json

# Load the Food Database
def load_csv(file_path=r"C:\Users\Meghana D Hegde\Downloads\disease_ingredient_harmful_amount_dataset.csv"):
    return pd.read_csv(file_path)

# Function to check harmful ingredients
def check_harmful_ingredients_csv(ingredients_list, user_profile, df):
    results = []

    # Iterate through user's medical conditions
    for disease in user_profile['diseases']:
        # Filter the DataFrame for the specific disease
        disease_data = df[df["Disease"].str.lower() == disease.lower()]

        # Check for harmful ingredients
        for _, row in disease_data.iterrows():
            harmful_ingredients = row["Harmful Ingredients"].split(", ")
            harmful_amount = row["Harmful Amount (grams/day)"]

            # Check for matches in the ingredients list
            for ingredient in ingredients_list:
                ingredient_lower = ingredient.lower()
                if any(harmful.lower() in ingredient_lower for harmful in harmful_ingredients):
                    results.append({
                        "Disease": disease,
                        "Harmful Ingredient": ingredient,
                        "Threshold (grams/day)": harmful_amount
                    })

    return results

@st.cache_resource
def load_food_model():
    return load_model('food_model.keras')

@st.cache_resource
def load_general_model():
    return EfficientNetB0(weights='imagenet')

food_model = load_food_model()
general_model = load_general_model()
def load_class_labels(file_path):
    with open(file_path, 'r') as f:
        food_labels = json.load(f)
    return food_labels

# Load the class labels from the saved JSON file
food_labels = load_class_labels('class_labels.json')
print("Class Labels:", food_labels)


# Step 3: Function to predict food from a new ima
def predict_food(image_path):
    # Load and preprocess the image for food model
    img = image.load_img(image_path, target_size=(224, 224))  # Load image and resize
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for EfficientNet

    # Make prediction using the fine-tuned food model
    predictions = food_model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)  # Index of highest prediction

    # Map the predicted index to the class label
    predicted_food_label = food_labels[predicted_class_idx[0]]
    food_confidence = predictions[0][predicted_class_idx[0]]  # Confidence score for the prediction

    return predicted_food_label, food_confidence  # Return two values

def predict_general_object(image_path):
    # Load and preprocess the image for general model (EfficientNetB0)
    img = image.load_img(image_path, target_size=(224, 224))  # Load image and resize
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for EfficientNet

    # Make prediction using the general-purpose model (EfficientNet)
    predictions = general_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode the top-3 predictions
    return decoded_predictions


import requests
import json
import re


# Function to call Gemini API
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyBCOUSy8DZ5Qzxhqq0gOkRCUJEHNVLjoZM")







def get_food_info_from_gpt2(food_name):
    try:
        # Call Gemini API for food information
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
        f"Provide detailed information about {food_name} . "
        f"Include nutritional information, ingredients, and allergens."
        f"It should be in this format NUTRITIONAL INFORMATION:the infoand in next line allergens and then ingredients with bold letters with emojis"
    )
        # Return the full response text from Gemini
        return response.text

    except Exception as e:
        return f"An error occurred while fetching food information: {str(e)}"


# Example usage



def food_scanner_page():
    st.title("Food Scanner")
    st.write("Upload an image of a food item to get detailed information.")

    # File uploader for image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image to a temporary file for processing
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Make prediction (assuming you have a function to predict the food item from the image)
        food_prediction, food_confidence = predict_food("temp_image.jpg")
        st.write(f"Food Prediction: The food item is *{food_prediction}* with confidence {food_confidence * 100:.2f}%")
        st.write("If the detected food is incorrect, please enter the correct name below:")
        manual_food_input = st.text_input("Enter food name manually (if needed):")

        # Use either the predicted food or the manually entered food name
        food_name_to_use = manual_food_input if manual_food_input else food_prediction
        if food_name_to_use:
            food_data = get_food_info_from_gpt2(food_name_to_use)
            if "error" in food_data:
                st.error(food_data["error"])
            else:
                st.subheader("Food Information")
                st.write(food_data)

        if food_confidence > 0.5:
            food_data = get_food_info_from_gpt2(food_prediction)
            st.json(food_data)

            if "error" in food_data:
                st.error(food_data["error"])
            else:
                st.subheader("Food Information")
                st.write(food_data)

                # Allergen matching with user profile
                if 'allergies' in st.session_state.user_profile:
                    user_allergies = set(st.session_state.user_profile['allergies'])
                    food_allergens = set(detect_allergens_in_ingredients(food_data["ingredients"]))

                    # Find intersection of user allergies and food allergens
                    risk_allergens = user_allergies & food_allergens
                    if risk_allergens:
                        st.warning(
                            f"⚠ Risk! Your profile indicates allergies to: {', '.join(risk_allergens)}. Please proceed with caution.")
        else:
            # Make general object prediction (for cases where food prediction isn't confident)
            general_predictions = predict_general_object("temp_image.jpg")
            st.write("General Object Predictions:")

            # Display the general object prediction results
            for i, (label, description, score) in enumerate(general_predictions):
                st.write(f"{i + 1}. *{description}* ({label}): {score * 100:.2f}%")

            # Use the top prediction from general object model to fetch food data
            top_prediction = general_predictions[0]  # top prediction
            food_data = get_food_info_from_gpt2(top_prediction[1])  # Use description as food item to get info

            st.subheader("Nutritional Information for the Predicted Food Item:")
            if "error" in food_data:
                st.error(food_data["error"])
            else:
                st.subheader("Food Information")
                st.write(food_data)

                # Check for allergens based on user profile
                if 'allergies' in st.session_state.user_profile:
                    user_allergies = set(st.session_state.user_profile['allergies'])
                    food_allergens = set(detect_allergens_in_ingredients(food_data["ingredients"]))

                    # Find intersection of user allergies and food allergens
                    risk_allergens = user_allergies & food_allergens
                    if risk_allergens:
                        st.warning(
                            f"⚠ Risk! Your profile indicates allergies to: {', '.join(risk_allergens)}. Please proceed with caution.")
    else:
        st.write("Please upload an image to get started.")


# Function to preprocess uploaded image for model prediction
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Function to load user profiles from the file
def load_profiles():
    if os.path.exists("user_profiles.json"):
        try:
            with open("user_profiles.json", "r") as f:
                profiles = [json.loads(line) for line in f]
            return {profile["username"]: profile for profile in profiles}
        except Exception as e:
            st.error(f"Error loading profiles: {e}")
            return {}
    return {}

# Function to save user profiles to a JSON file
def save_profile(profile):
    profiles = load_profiles()
    profiles[profile["username"]] = profile  # Add/Update the profile

    try:
        with open("user_profiles.json", "w") as f:
            for profile in profiles.values():
                json.dump(profile, f)
                f.write("\n")
    except Exception as e:
        st.error(f"Error saving profile: {e}")

# Function to fetch nutritional, allergen, and ingredients data from Open Food Facts API
def fetch_food_data(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Failed to fetch data from Open Food Facts API."}

    data = response.json()

    if "product" in data and data["product"]:
        product = data["product"]

        # Try to extract nutritional information from the response
        nutrition = product.get("nutriments", {})
        allergens = product.get("allergens", "No allergens listed.")

        # Attempt to get ingredients from multiple possible fields
        ingredients = product.get("ingredients_text", "")
        if not ingredients:
            ingredients = product.get("ingredients_text_tags", "")
        if not ingredients:
            ingredients = product.get("ingredients_text_fr", "")
        if not ingredients:
            ingredients = "No ingredients found."

        # Format the nutritional information
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

# Function to detect allergens in ingredients
def detect_allergens_in_ingredients(ingredients):
    allergens_keywords = [
        "peanut", "milk", "gluten", "egg", "soy", "wheat", "fish", "shellfish",
        "nuts", "sesame", "lactose", "almond", "cashew", "hazelnut", "walnut",
        "oats", "mustard", "celery", "sulphur", "corn", "yeast", "mollusks",
        "barley", "rye", "spelt", "kamut", "triticale", "buckwheat", "chickpea",
        "lentils", "peas", "beans", "apple", "carrot", "potato", "tomato",
        "orange", "grape", "banana", "mango", "papaya", "kiwi", "pineapple",
        "coconut", "cocoa", "chocolate", "sugar", "rice", "chestnut", "pecan",
        "brazil nut", "macadamia", "pine nut", "pistachio", "hazelnut", "cashew",
        "sulfites", "glutamate", "artificial color", "artificial flavor",
        "tartrazine", "msg", "phthalates", "peach", "pear", "apricot", "plum",
        "fig", "pomegranate", "blackberry", "strawberry", "blueberry", "raspberry",
        "cabbage", "broccoli", "cauliflower", "spinach", "lettuce", "eggplant",
        "zucchini", "asparagus", "cucumber", "bell pepper", "pumpkin", "squash",
        "cabbage", "onion", "garlic", "ginger", "turmeric", "caraway", "oregano",
        "parsley", "basil", "rosemary", "sage", "thyme", "cumin", "curry",
        "chili pepper", "saffron", "vanilla", "anise", "fennel", "cinnamon",
        "nutmeg", "cloves", "cardamom", "bay leaf", "dill", "tarragon", "marjoram",
        "coriander", "fennel", "szechuan pepper", "chili", "black pepper", "white pepper"
    ]

    # Normalize the ingredients and allergens to lowercase to ensure case-insensitive matching
    ingredients_lower = ingredients.lower()
    allergens_found = []

    for allergen in allergens_keywords:
        if allergen in ingredients_lower:
            allergens_found.append(allergen.capitalize())

    return allergens_found


# Set up Streamlit app title
st.title("NUTRISHIELD - Nutrition and Health Assistant")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the app mode", ["Home", "User Profile", "Barcode Scanner","Food Scanner","Chatbot","Diet Plans"])

# Home Screen
if app_mode == "Home":
    st.header("Welcome to NUTRISHIELD")
    st.write("Navigate to the User Profile or Barcode Scanner to get started.")

# User Profile Screen
elif app_mode == "User Profile":
    st.header("User Profile")
    st.write("Enter your information below:")

    # Check if the user is logged in (based on session_state)
    if st.session_state.get("user_profile"):
        # Allow user to view and update their existing profile
        st.write("Your current profile:")
        st.json(st.session_state.user_profile)
    else:
        st.write("Create a new profile:")

    # Input fields for user profile
    username = st.text_input("Username", value=st.session_state.user_profile.get("username", "") if st.session_state.get("user_profile") else "")
    age = st.number_input("Age", min_value=1, step=1, value=st.session_state.user_profile.get("age", 1))
    weight = st.number_input("Weight (kg)", min_value=1.0, value=st.session_state.user_profile.get("weight", 1.0))
    height = st.number_input("Height (cm)", min_value=1.0, value=st.session_state.user_profile.get("height", 1.0))

    # Medical records section
    st.subheader("Medical Record")
    allergies_input = st.text_area("Allergies (comma separated)", value=", ".join(st.session_state.user_profile.get("allergies", [])))
    diseases_input = st.text_area("Diseases (comma separated)", value=", ".join(st.session_state.user_profile.get("diseases", [])))

    if st.button("Submit"):
        if username and age and weight and height:
            # Save user profile to session_state
            st.session_state.user_profile = {
                "username": username,
                "age": age,
                "weight": weight,
                "height": height,
                "allergies": [allergy.strip().lower() for allergy in allergies_input.split(',')],
                "diseases": [disease.strip().lower() for disease in diseases_input.split(',')]
            }
            save_profile(st.session_state.user_profile)
            st.success(f"Profile created/updated: {st.session_state.user_profile}")
        else:
            st.error("All fields are required!")

# Barcode Scanner Screen
# Barcode Scanner Screen
elif app_mode == "Barcode Scanner":
    st.header("Barcode Scanner")
    st.write("Capture an image to scan the barcode. You can either use your camera or upload an image.")

    # Option to use webcam or upload an image
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Capture image using camera
    img_file = st.camera_input("Take a picture to scan the barcode")

    # Handle both uploaded image and camera input
    img = None
    if uploaded_img:
        # If user uploaded an image, convert it to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.write("Image uploaded.")

    elif img_file:
        # If user used the camera, convert it to OpenCV format
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.write("Image captured from camera.")

    if img is not None:
        # Show the image
        st.image(img, caption="Scanned Image", use_column_width=True)

        # Decode the barcode
        decoded_objects = decode(img)
        if decoded_objects:
            barcode_data = decoded_objects[0].data.decode("utf-8")
            st.success(f"Decoded Barcode: {barcode_data}")

            # Fetch nutritional, allergen, and ingredients data
            food_data = fetch_food_data(barcode_data)

            if "error" in food_data:
                st.error(food_data["error"])
            else:
                # Display Nutritional Information
                st.subheader("🍽 Nutritional Information:")
                st.write(f"🔥 Energy (kcal): {food_data['nutritional_info']['energy_kcal']}")
                st.write(f"🥚 Fat: {food_data['nutritional_info']['fat']} g")
                st.write(f"🥚 Saturated Fat: {food_data['nutritional_info']['saturated_fat']} g")
                st.write(f"🍞 Carbohydrates: {food_data['nutritional_info']['carbohydrates']} g")
                st.write(f"🍬 Sugars: {food_data['nutritional_info']['sugars']} g")
                st.write(f"🍠 Fiber: {food_data['nutritional_info']['fiber']} g")
                st.write(f"🍗 Proteins: {food_data['nutritional_info']['proteins']} g")
                st.write(f"🧂 Salt: {food_data['nutritional_info']['salt']} g")
                st.markdown("---")

                # Allergens check based on user profile
                allergens_from_ingredients = detect_allergens_in_ingredients(food_data["ingredients"])

                # Extract allergens from the product's allergens field
                if food_data["allergens"] != "No allergens listed.":
                    allergens_from_ingredients += [allergen.strip().lower() for allergen in food_data["allergens"].split(',')]
                allergens_from_ingredients = list(set(allergens_from_ingredients))  # Remove duplicates

                # Debugging: Display allergens found in the food data
                st.write("Allergens found in the food data:", allergens_from_ingredients)

                st.subheader("⚠ Allergens:")
                if allergens_from_ingredients:
                    found_allergens = ", ".join(allergens_from_ingredients)
                    st.warning(f"⚠ Allergen Warning: Contains allergens: {found_allergens}")
                # Health Risk Alert based on user profile allergies
                # Health Risk Alert based on user profile allergies
                if 'allergies' in st.session_state.user_profile:
                    user_allergies = set(st.session_state.user_profile['allergies'])
                    # Make sure the user allergies are lowercased for case-insensitive comparison
                    user_allergies = {allergy.strip().lower() for allergy in user_allergies}

                    # Debugging: Display user profile allergies
                    st.write("User allergies (profile):", user_allergies)

                    # Strip and lowercase food allergens as well
                    allergens_from_ingredients = [allergen.strip().lower() for allergen in allergens_from_ingredients]

                    # Debugging: Display allergens extracted from the food
                    st.write("Allergens extracted from food:", allergens_from_ingredients)

                    # Find the intersection between user allergies and food allergens
                    risk_allergens = user_allergies & set(allergens_from_ingredients)

                    # Debugging: Display the intersection of user allergies and food allergens
                    st.write("Risk allergens found:", risk_allergens)

                    # If there are any risk allergens, show the warning
                    if risk_allergens:
                        risk_allergens_str = ', '.join(risk_allergens)
                        st.markdown(
                            f"""
                                    <div style="background-color: #FFCCCB; border-radius: 10px; padding: 20px; border: 2px solid red; font-size: 20px; color: red; font-weight: bold;">
                                        ⚠ <b>Risk Alert:</b> Your medical record indicates allergies to <span style="color: darkred;">{risk_allergens_str}</span>. <br> <b>Please proceed with caution!</b>
                                    </div>
                                    """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("No risk allergens found.")
                    csv_file_path = r"C:\Users\Meghana D Hegde\Downloads\disease_ingredient_harmful_amount_dataset.csv"  # Replace with your CSV file path
                    health_risks_df = load_csv(csv_file_path)

                    # Check for harmful ingredients
                    harmful_results = check_harmful_ingredients_csv(food_data["ingredients"].split(", "),
                                                                    st.session_state.user_profile, health_risks_df)

                    if harmful_results:
                        st.subheader("⚠ Health Risk Check:")
                        for result in harmful_results:
                            st.warning(
                                f"Disease: {result['Disease']} - Harmful Ingredient: {result['Harmful Ingredient']} exceeds the threshold of {result['Threshold (grams/day)']} grams/day.")
                    else:
                        st.write("No harmful ingredients detected based on your medical conditions.")

                # Display Ingredients Information
                st.subheader("🍽 Ingredients Information:")
                st.write(food_data["ingredients"])

        else:
            st.warning("No barcode detected. Try again with a clearer image.")
elif app_mode == "Food Scanner":
    food_scanner_page()

elif app_mode == "Chatbot":
    st.title("HealthyBuddy 🥗🤗")
    st.markdown("Meet Foodie! Ask any food-related question, and our Foodie will assist you.")
    chatbot_url = "https://cdn.botpress.cloud/webchat/v2.2/shareable.html?configUrl=https://files.bpcontent.cloud/2024/12/07/18/20241207185412-QBTY0H6R.json"
    components.iframe(chatbot_url, width=800, height=600)

elif app_mode == "Diet Plans":
    # Assuming `recipe_model.pkl` is loaded as in the previous example
    model_file_path = "recipe_model.pkl"
    with open(model_file_path, 'rb') as model_file:
        similarity_matrix, recipe_data, vectorizer, svd = pickle.load(model_file)


    # Function to classify dietary preferences based on ingredients
    def classify_dietary_preference(ingredients, dietary_preference):
        ingredients = " ".join(ingredients).lower()

        # Define keywords for classification
        vegetarian_exclude_keywords = ["meat", "chicken", "fish", "mutton", "pork", "lamb", "bacon", "turkey", "beef",
                                       "duck", "sausage", "ham", "goat", "veal", "game", "rabbit", "venison",
                                       "shellfish"]
        vegan_exclude_keywords = ["meat", "chicken", "fish", "pork", "lamb", "bacon", "turkey", "cheese", "milk",
                                  "butter",
                                  "eggs", "honey", "yogurt", "cream", "gelatin"]
        non_vegetarian_keywords = ["chicken", "fish", "pork", "lamb", "bacon", "turkey", "beef", "duck", "sausage",
                                   "ham",
                                   "goat", "veal", "game", "rabbit", "venison", "shellfish", "shrimp", "crab",
                                   "lobster"]

        if dietary_preference == "Non-Vegetarian":
            if any(keyword in ingredients for keyword in non_vegetarian_keywords):
                return True
            else:
                return False

        if dietary_preference == "Vegetarian":
            if any(keyword in ingredients for keyword in vegetarian_exclude_keywords):
                return False
            else:
                return True

        if dietary_preference == "Vegan":
            if any(keyword in ingredients for keyword in vegan_exclude_keywords):
                return False
            else:
                return True

        return True


    # Filter recipes based on user input
    def filter_recipes(region, allergens, dietary_preference):
        # Filter by region
        filtered_recipes = recipe_data[recipe_data['region'].str.contains(region, case=False, na=False)]

        # Filter by allergens
        if allergens:
            for allergen in allergens:
                filtered_recipes = filtered_recipes[~filtered_recipes['ingredients'].apply(
                    lambda x: any(allergen.lower() in ingredient.lower() for ingredient in eval(x)))]

        # Filter by dietary preference
        if dietary_preference != "Any":
            filtered_recipes['dietary_preference'] = filtered_recipes['ingredients'].apply(
                lambda x: classify_dietary_preference(eval(x), dietary_preference))
            filtered_recipes = filtered_recipes[filtered_recipes['dietary_preference'] == True]

        return filtered_recipes


    # Main diet plan generation function
    def generate_diet_plan(region_input, allergen_input, dietary_preference_input):
        st.subheader("Recommended Recipes for Your Diet Plan:")

        # Filter recipes based on user input
        filtered_recipes = filter_recipes(region_input, allergen_input, dietary_preference_input)

        if filtered_recipes.empty:
            st.write("No recipes found based on your criteria.")
        else:
            for _, recipe in filtered_recipes.iterrows():
                st.write(f"**{recipe['name']}**")
                st.write(f"**Region:** {recipe['region']}")
                st.write(f"**Ingredients:** {recipe['ingredients']}")
                st.write(f"**Dietary Preference:** {recipe.get('dietary_preference', 'Unknown')}")
                st.write(f"---")


    # Streamlit UI setup
   
    st.title("Diet Plan Generator 🍽️")

    # Sidebar for user input
    st.sidebar.header("Input Criteria")

    regions = ['Fusion', 'France', 'Greece', 'Italy', 'Korea', 'Mediterranean', 'Mexico', 'Spain', 'Thailand',
               'Vietnam',
               'Japan', 'India', 'China', 'Africa', 'USA']
    region_input = st.sidebar.selectbox("Select region", regions)

    allergen_input = st.sidebar.multiselect("Select allergens to avoid",
                                            ["Gluten", "Dairy", "Nuts", "Shellfish", "Soy", "Eggs"])
    dietary_preference_input = st.sidebar.selectbox("Select dietary preference",
                                                    ["Any", "Vegetarian", "Non-Vegetarian", "Vegan"])

    # Button to trigger diet plan generation
    if st.sidebar.button("Generate Diet Plan"):
        generate_diet_plan(region_input, allergen_input, dietary_preference_input)
