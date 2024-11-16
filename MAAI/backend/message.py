import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# Convert the dataframe rows to text format for embedding
def row_to_text(row):
    text = f"This is a {row['AGE']} year old "
    text += "male" if row['GENDER'] == 1 else "female"
    
    conditions = []
    if row['SMOKING'] == 1:
        conditions.append("smokes")
    if row['ALCOHOL CONSUMING'] == 1:
        conditions.append("drinks alcohol")
    if row['YELLOW_FINGERS'] == 1:
        conditions.append("has yellow fingers")
    if row['ANXIETY'] == 1:
        conditions.append("has anxiety")
    if row['PEER_PRESSURE'] == 1:
        conditions.append("experiences peer pressure")
    if row['CHRONIC DISEASE'] == 1:
        conditions.append("has chronic disease")
    if row['WHEEZING'] == 1:
        conditions.append("experiences wheezing")
    if row['COUGHING'] == 1:
        conditions.append("has coughing")
    if row['SHORTNESS OF BREATH'] == 1:
        conditions.append("has shortness of breath")
    if row['SWALLOWING DIFFICULTY'] == 1:
        conditions.append("has difficulty swallowing")
    if row['CHEST PAIN'] == 1:
        conditions.append("has chest pain")
    
    if conditions:
        text += " who " + ", ".join(conditions)
    
    text += ". Lung cancer: "
    text += "Yes" if row['LUNG_CANCER'] == 1 else "No"
    
    return text

# Create text representations of all rows
documents = df.apply(row_to_text, axis=1).tolist()

# Generate embeddings for all documents
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)

# Save embeddings and documents
embeddings_np = np.array(doc_embeddings).astype('float32')
with open('survey_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'embeddings': embeddings_np,
        'documents': documents
    }, f)

# Update query_embeddings function to use survey data
def query_embeddings(query, top_k=3):
    # Load saved embeddings and documents
    with open('survey_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
        documents = data['documents']
    
    # Encode the query
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query]).astype('float32')

    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)
    most_similar_idx = similarities.argsort()[0][-top_k:][::-1]

    # Return similar cases
    similar_docs = [(documents[i], similarities[0][i]) for i in most_similar_idx]
    return similar_docs

# Function to perform search based on user query
def process_user_query(user_message: str):
    top_docs = query_embeddings(user_message)
    
    # Check if best match is below 90%
    best_similarity = top_docs[0][1] * 100
    if best_similarity < 90:
        response = "I need more information to find relevant cases. Please provide details about:\n\n"
        response += "• Your gender (male/female)\n"
        response += "• Any symptoms you're experiencing (like coughing, chest pain, breathing issues)\n"
        response += "• Other relevant factors (anxiety, peer pressure, chronic diseases)\n\n"
        response += "(low confidence):\n\n"
    else:
        response = "Here are the most similar cases from our database:\n\n"
    
    # Add matching cases
    # for doc, sim in top_docs:
    #     response += f"\n"
    
    return response


# ef call_groq(prompt: str) -> str:
#     """
#     Call Groq API to generate a response based on the given prompt.
    
#     Args:
#         prompt (str): The prompt to send to Groq
        
#     Returns:
#         str: Generated response from Groq, or a fallback message if the API call fails
#     """
#     # Load environment variables
#     load_dotenv()
    
#     # Get API key from environment variables
#     api_key = "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"
#     if not api_key:
#         return "Error: GROQ_API_KEY not found in environment variables."

#     # API endpoint
#     url = "https://api.groq.com/v1/chat/completions"
    
#     # Request headers
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
    
#     # Request payload
#     data = {
#         "model": "mixtral-8x7b-32768",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful medical assistant. Keep responses concise and friendly."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "temperature": 0.7,
#         "max_tokens": 500
#     }
    
#     try:
#         # Make the API call with a timeout
#         with httpx.Client(timeout=30.0) as client:
#             response = client.post(url, headers=headers, json=data)
#             response.raise_for_status()  # Raise an exception for bad status codes
            
#             # Parse the response
#             result = response.json()
#             return result['choices'][0]['message']['content']
            
#     except httpx.TimeoutException:
#         return "I apologize, but the response took too long. Please try again."
#     except httpx.HTTPError as e:
#         return f"An error occurred while communicating with the API: {str(e)}"
#     except json.JSONDecodeError:
#         return "Error: Received invalid response from the API."
#     except KeyError:
#         return "Error: Unexpected response format from the API."
#     except Exception as e:
#         return f"An unexpected error occurred: {str(e)}"