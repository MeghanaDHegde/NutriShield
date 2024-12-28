import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# Log starting point
print("Script started.", flush=True)

# Load the Recipe Database
recipe_file_path = r"C:\Users\Meghana D Hegde\Downloads\Filtered_Recepies.xlsx"
print("Loading recipe database...", flush=True)

try:
    recipe_data = pd.read_excel(recipe_file_path, sheet_name='Sheet1')
    print(f"Recipe database loaded successfully with {recipe_data.shape[0]} recipes.", flush=True)
except FileNotFoundError:
    print("Error: The specified file was not found.", flush=True)
    raise
except Exception as e:
    print(f"Error loading file: {e}", flush=True)
    raise

# Preprocess Recipe Data
print("Preprocessing recipe data...", flush=True)
try:
    recipe_data.columns = recipe_data.columns.str.strip().str.lower()
    recipe_data = recipe_data.dropna(subset=['name', 'ingredients', 'region'])
    print(f"Data after preprocessing: {recipe_data.shape[0]} recipes loaded.", flush=True)
except Exception as e:
    print(f"Error during preprocessing: {e}", flush=True)
    raise

# Sample a subset of the recipes (for example, 10,000 recipes) to reduce memory usage
sample_size = 10000
sampled_data = recipe_data.sample(n=sample_size, random_state=42)
print(f"Sampled {sample_size} recipes for similarity calculation.", flush=True)

# Combine all textual information for similarity
print("Combining ingredients and region into a single feature...", flush=True)
sampled_data['combined_features'] = sampled_data['ingredients'].str.lower() + " " + sampled_data['region'].str.lower()
print("Combined features created.", flush=True)

# Extract unique regions from the sampled data
regions = sampled_data['region'].unique()
print("Unique regions extracted:", regions)

# Convert textual data into vectors using CountVectorizer
print("Converting textual data into vectors...", flush=True)
vectorizer = CountVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(sampled_data['combined_features'])
print("Vectorization complete.", flush=True)

# Apply dimensionality reduction (optional, to reduce memory usage)
from sklearn.decomposition import TruncatedSVD

print("Applying dimensionality reduction...", flush=True)
svd = TruncatedSVD(n_components=100, random_state=42)
reduced_vectors = svd.fit_transform(feature_vectors)
print("Dimensionality reduction complete. Reduced to 100 components.", flush=True)

# Compute similarity matrix using batch processing
print("Computing similarity matrix in batches...", flush=True)

batch_size = 1000  # You can adjust this based on your memory limitations
num_samples = len(sampled_data)
similarity_matrix = np.zeros((num_samples, num_samples))

for i in range(0, num_samples, batch_size):
    end_idx = min(i + batch_size, num_samples)
    batch_vectors = reduced_vectors[i:end_idx]

    # Compute the similarity of this batch with all other vectors
    similarity_batch = cosine_similarity(batch_vectors, reduced_vectors)
    similarity_matrix[i:end_idx] = similarity_batch

print("Similarity matrix computation complete.", flush=True)

# Save the model and required data
model_file_path = r"C:\Users\Meghana D Hegde\PycharmProjects\pythonProject1\recipe_model.pkl"
print("Saving the model and data...", flush=True)

try:
    with open(model_file_path, 'wb') as model_file:
        pickle.dump((similarity_matrix, sampled_data, vectorizer, svd), model_file)
    print("Model trained and saved successfully.", flush=True)
except Exception as e:
    print(f"Error saving the model: {e}", flush=True)
    raise
