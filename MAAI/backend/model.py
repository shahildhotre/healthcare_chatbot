import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
from chromadb.utils import embedding_functions
import hashlib

class NugenEmbeddingFunction():
    def __init__(self, api_token: str, dimensions: int = 123):
        self.api_url = "https://api.nugen.in/inference/embeddings"
        self.api_token = api_token
        self.dimensions = dimensions
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        for text in texts:
            payload = {
                "input": text,
                "model": "nugen-flash-embed",
                "dimensions": self.dimensions
            }
            
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=self.headers
                )
# Raise exception for bad status codes
                response.raise_for_status()
# Assuming API returns embedding array
                embedding = response.json()
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: {str(e)}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimensions)
        
        return embeddings

class CSVRAGBot:
    def __init__(self, csv_path: str, text_columns: List[str], api_token: str):
        """
        Initialize RAG bot for CSV files
        
        Args:
            csv_path: Path to the CSV file
            text_columns: List of column names containing text to be processed
        """
        self.csv_path = csv_path
        self.text_columns = text_columns

        self.embedding_function = NugenEmbeddingFunction(api_token=api_token)
        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.db.create_collection(
            name="csv_collection",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-mpnet-base-v2"
            )
        )
    
    def load_csv(self) -> pd.DataFrame:
        """Load CSV file into pandas DataFrame"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise
    
    def process_row(self, row: pd.Series) -> str:
        """Process a single row into a text string"""
        text_parts = []
        # Add column names for context
        for col in self.text_columns:
            if col in row:
                text_parts.append(f"{col}: {str(row[col])}")
        
        # Combine all text columns
        return " | ".join(text_parts)
    
    def generate_row_id(self, row: pd.Series, index: int) -> str:
        """Generate a unique ID for each row"""
        # Create a string combining index and key columns
        row_string = f"{index}-{'-'.join(str(row[col]) for col in self.text_columns)}"
        # Generate a hash for the ID
        return hashlib.md5(row_string.encode()).hexdigest()
    
    def process_csv(self) -> List[Dict]:
        """Process CSV file into documents"""
        df = self.load_csv()
        
        # Text splitter for long text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        processed_docs = []
        
        for index, row in df.iterrows():
            # Process row into text
            text = self.process_row(row)
            
            # Generate unique ID
            doc_id = self.generate_row_id(row, index)
            
            # Create metadata (store all non-text columns)
            metadata = {
                'row_index': index,
                'source': self.csv_path
            }
            for col in df.columns:
                if col not in self.text_columns:
                    metadata[col] = str(row[col])
            
            # Split if text is too long
            if len(text) > 1000:
                splits = text_splitter.split_text(text)
                for i, split in enumerate(splits):
                    split_id = f"{doc_id}-{i}"
                    processed_docs.append({
                        'id': split_id,
                        'text': split,
                        'metadata': {**metadata, 'chunk': i}
                    })
            else:
                processed_docs.append({
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata
                })
        
        print(f"Processed {len(processed_docs)} documents")
        return processed_docs
    
    def add_to_vectorstore(self, processed_docs: List[Dict]):
        """Add documents to vector store"""
        try:
            self.collection.add(
                ids=[doc['id'] for doc in processed_docs],
                documents=[doc['text'] for doc in processed_docs],
                metadatas=[doc['metadata'] for doc in processed_docs]
            )
            print(f"Added {len(processed_docs)} documents to vector store")
        except Exception as e:
            print(f"Error adding to vector store: {str(e)}")
            raise
    
    def query(self, query_text: str, n_results: int = 3):
        """Query the vector store"""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error querying vector store: {str(e)}")
            raise

# Example usage
def main():
    # Example configuration
    csv_path = "./dataset.csv"
    text_columns = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", 
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", 
    "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN", "LUNG_CANCER"
]
  # Replace with your text column names
    api_token = 'nugen-z6v18EFklagYS_4ZGMC80w'
    # Initialize RAG bot
    rag_bot = CSVRAGBot(csv_path=csv_path, text_columns=text_columns,
                        api_token = api_token)
    
    # Process CSV and add to vector store
    processed_docs = rag_bot.process_csv()
    print(processed_docs)
    rag_bot.add_to_vectorstore(processed_docs)
    
    # Test query
    query = "your test query"
    results = rag_bot.query(query)
    print("\nQuery Results:")
    print(results)

if __name__ == "__main__":
    main()
