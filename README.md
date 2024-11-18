# AI-Powered Healthcare Chatbot  

**An Intelligent Assistant for Symptom Analysis and Disease Prediction**  

## Overview  

This project was developed as part of the **GMG (Garje Marathi Global) – MAAI Hackathon**, where our team secured 1st place! The AI-powered healthcare chatbot is designed to analyze symptoms provided by users and offer insights into potential diseases, with a focus on **early detection of lung cancer** in this prototype.  

### Key Features  
- **Symptom Analysis:** Users can input symptoms, and the chatbot responds with probabilities (high/low) of having a disease.  
- **Advanced AI Integration:** Utilizes **Nugen API** for generating embeddings and implementing **cosine similarity** for document retrieval.  
- **Improved Healthcare Accessibility:** Cost-effective solution for patients and healthcare providers to improve early diagnosis, engagement, and satisfaction.  
- **Streamlit UI:** Interactive user interface for seamless interaction.  

---

## Project Workflow  

1. **Load Data:** Import lung cancer survey data from [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer).  
2. **Preprocess Data:**  
   - Encode categorical variables.  
   - Fill missing data.  
3. **Train Classifier:** Use a **Decision Tree Classifier** to predict lung cancer.  
4. **Generate Text Data:** Convert data rows into natural language descriptions.  
5. **Embed Document Texts:** Utilize **Nugen API** for embedding creation.  
6. **Store Embeddings:** Save embeddings in a pickle file for efficient querying.  
7. **User Query Input:** Allow users to input queries about symptoms or diseases.  
8. **Query Embedding:** Convert user queries into embeddings using **Nugen API**.  
9. **Cosine Similarity Computation:** Compare query embeddings with pre-stored embeddings for relevance ranking.  
10. **Retrieve and Rank:** Identify and rank top-k relevant documents based on similarity scores.  
11. **Chatbot Response:** Provide users with insights on their probability of having lung cancer (High/Low).  

---

## Technology Stack  

- **Programming Language:** Python  
- **APIs and Libraries:** Nugen API, Pandas, Scikit-learn, Streamlit  
- **Machine Learning Models:** Decision Tree Classifier, SVM, Random Forest, KNN, BERT (for advanced use cases)  
- **Deployment:** Oracle Cloud  

---

## Getting Started  

### Prerequisites  
1. Install Python 3.8 or above.  
2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```  

### Running the Application  

#### Frontend  
To start the Streamlit user interface, run:  
```bash  
streamlit run streamlit_app.py  
```  

#### Backend  
To launch the backend API using Uvicorn, run:  
```bash  
uvicorn main:app --reload  
```  

Ensure both the frontend and backend are running simultaneously for the application to function properly.  

---

## Team  

- **Anupama Aphale**  
- **Chirag Dhamange**  
- **Shahil Dhotre**  
- **Shubham Narkhede**  
- **Rahul Phadtare**  
- **Savani Shrotri**  

---

## Acknowledgments  

- **Garje Marathi Global** and its visionary founder **Anand Ganu**, along with his dedicated team, for organizing this remarkable event.  
- **Stanford University** for hosting the hackathon.  
- **Pushkar Nandkar** (SambaNova Systems) and **Saurabh Netravalkar** (Oracle) for their invaluable contributions as sponsors and mentors.  
- **Kaustubh Supekar** (Stanford University) and **Niraj Kumar Singh** (Nugen) for their exceptional guidance.  

---