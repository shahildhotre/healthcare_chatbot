import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the Kaggle dataset
df = pd.read_csv("survey lung cancer.csv")

# Preprocess all categorical variables
categorical_mappings = {
    'GENDER': {'Male': 1, 'Female': 0},
    'SMOKING': {'YES': 1, 'NO': 0},
    'YELLOW_FINGERS': {'YES': 1, 'NO': 0},
    'ANXIETY': {'YES': 1, 'NO': 0},
    'PEER_PRESSURE': {'YES': 1, 'NO': 0},
    'CHRONIC DISEASE': {'YES': 1, 'NO': 0},
    'WHEEZING': {'YES': 1, 'NO': 0},
    'ALCOHOL CONSUMING': {'YES': 1, 'NO': 0},
    'COUGHING': {'YES': 1, 'NO': 0},
    'SHORTNESS OF BREATH': {'YES': 1, 'NO': 0},
    'SWALLOWING DIFFICULTY': {'YES': 1, 'NO': 0},
    'CHEST PAIN': {'YES': 1, 'NO': 0},
    'LUNG_CANCER': {'YES': 1, 'NO': 0}
}

# Apply mappings to categorical columns
for column, mapping in categorical_mappings.items():
    if column in df.columns:
        # Convert column to string type first
        df[column] = df[column].astype(str)
        # Convert to uppercase and map
        df[column] = df[column].str.upper().map(mapping)
        # Fill any NaN values with 0
        df[column] = df[column].fillna(0)

# Make sure AGE is numeric
if 'AGE' in df.columns:
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
    df['AGE'] = df['AGE'].fillna(df['AGE'].mean())

# Select relevant features and target
X = df[[
    "GENDER", 
    "AGE", 
    "SMOKING", 
    "YELLOW_FINGERS", 
    "ANXIETY", 
    "PEER_PRESSURE", 
    "CHRONIC DISEASE",
    "WHEEZING", 
    "ALCOHOL CONSUMING", 
    "COUGHING", 
    "SHORTNESS OF BREATH", 
    "SWALLOWING DIFFICULTY", 
    "CHEST PAIN"
]]
y = df['LUNG_CANCER']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Model evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))



# Load pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example lung cancer-related queries and documents
queries = ["What are the symptoms of lung cancer?", "What causes lung cancer?"]
documents = [
    "Lung cancer symptoms include persistent cough, chest pain, and shortness of breath.",
    "Lung cancer is caused mainly by smoking, exposure to radon, and asbestos."
]

# Generate embeddings
query_embeddings = model.encode(queries)
doc_embeddings = model.encode(documents)

# Example: Find the most similar document to the first query
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([query_embeddings[0]], doc_embeddings)
print("Most similar document to the first query:", documents[np.argmax(similarity)])


# Load pre-trained SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example lung cancer-related queries and documents
queries = ["What are the symptoms of lung cancer?", "What causes lung cancer?"]
documents = [
    "Lung cancer symptoms include persistent cough, chest pain, and shortness of breath.",
    "Lung cancer is caused mainly by smoking, exposure to radon, and asbestos."
]

# Generate embeddings
query_embeddings = model.encode(queries)
doc_embeddings = model.encode(documents)

# Example: Find the most similar document to the first query
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([query_embeddings[0]], doc_embeddings)
print("Most similar document to the first query:", documents[np.argmax(similarity)])

# Convert embeddings into a numpy array (which can be stored)
embeddings_np = np.array(query_embeddings).astype('float32')

# Save embeddings to a file (e.g., .npy or .pkl)
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_np, f)

# Load saved embeddings and documents
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)


# Function to perform search based on user query
def query_embeddings(query, top_k=3):
    # Encode the query into an embedding using the same model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query]).astype('float32')

    # Compute cosine similarity between query embedding and stored embeddings
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get indices of the top-k most similar documents
    most_similar_idx = similarities.argsort()[0][-top_k:][::-1]  # Sort and get top-k results

    # Retrieve the top-k most similar documents
    similar_docs = [(documents[i], similarities[0][i]) for i in most_similar_idx]
    
    return similar_docs

# Function to perform search based on user query
def process_user_query(user_message: str):
    # The user_message is now directly the query string
    # Get similar documents
    print("User message before looking for embeddings:", user_message)
    top_docs = query_embeddings(user_message)
    
    # Format the response
    response = "Here are the most relevant matches:\n\n"
    for doc, sim in top_docs:
        response += f"• {doc} (Similarity: {sim:.4f})\n"
    
    return response
