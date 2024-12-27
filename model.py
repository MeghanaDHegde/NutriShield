import pandas as pd

# Expanded dataset with harmful amounts in grams
data = {
    "Disease": [
        "Diabetes",
        "Hypertension",
        "Celiac Disease",
        "Chronic Kidney Disease",
        "Heart Disease",
        "Lactose Intolerance",
        "Irritable Bowel Syndrome (IBS)",
        "Gout",
        "Liver Disease",
        "Osteoporosis",
        "Cancer (General)"
    ],
    "Harmful Ingredients": [
        "Sugar, Refined Carbs, High-Fructose Corn Syrup",
        "Salt, Caffeine, Alcohol",
        "Gluten (Wheat, Barley, Rye)",
        "Sodium, Phosphorus Additives, High-Protein Foods",
        "Saturated Fats, Trans Fats, Red Meat",
        "Milk, Dairy Products, Lactose Additives",
        "Spicy Foods, High-Fat Foods, Artificial Sweeteners",
        "High-Purine Foods (Organ Meats, Seafood), Alcohol",
        "Alcohol, Processed Foods, Sugary Drinks",
        "Excessive Salt, Caffeine, Carbonated Drinks",
        "Processed Meats, Sugary Foods, Alcohol, Trans Fats"
    ],
    "Harmful Amount (grams/day)": [
        25,   # Diabetes (added sugar threshold)
        6,    # Hypertension (salt threshold)
        1,    # Celiac Disease (gluten threshold, trace amounts)
        2,    # Chronic Kidney Disease (sodium threshold)
        20,   # Heart Disease (saturated fat threshold)
        10,   # Lactose Intolerance (lactose tolerance)
        15,   # IBS (artificial sweeteners threshold)
        300,  # Gout (purines threshold in food)
        10,   # Liver Disease (alcohol threshold)
        2,    # Osteoporosis (excessive salt threshold)
        50    # Cancer (processed meats threshold)
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file = r"C:\Users\Meghana D Hegde\Downloads\disease_ingredient_harmful_amount_dataset.csv"
df.to_csv(csv_file, index=False)

# Display the DataFrame
print("Dataset saved to", csv_file)
print(df)
