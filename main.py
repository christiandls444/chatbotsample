import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
dataset_path = "chatbot.xlsx"  # Update with the actual path
df = pd.read_excel(dataset_path)

# Convert questions to string
df['questions'] = df['questions'].astype(str)

# Preprocess data
df['processed_question'] = df['questions'].apply(lambda x: x.lower())

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the questions
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_question'])

def get_response(query):
    query_vector = tfidf_vectorizer.transform([query.lower()])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    
    best_match_index = similarity_scores.argmax()
    best_answer = df['answers'][best_match_index]
    return best_answer

# Streamlit app
def main():
    st.title("Egypt Tourist Chatbot")

    # Initialize chat history
    chat_history = []

    user_query = st.text_input("You:", "How do I obtain an Egyptian tourist visa?")
    if user_query:
        chat_history.append(("You", user_query))
        chatbot_response = get_response(user_query)
        chat_history.append(("Chatbot", chatbot_response))

    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for sender, message in chat_history:
            if sender == "You":
                st.write("You:", message)
            else:
                st.write("Chatbot:", message)

if __name__ == "__main__":
    main()
