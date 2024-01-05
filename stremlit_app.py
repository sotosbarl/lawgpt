from transformers import BertModel, BertTokenizer , pipeline, AutoModelForSeq2SeqLM, DebertaV2Tokenizer, AutoModelForQuestionAnswering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import streamlit as st
import pickle



# Define the multilingual model name
model_name = 'bert-base-multilingual-cased'  # Multilingual BERT model

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)



with open('articles_list.pkl', 'rb') as file:
    articles_list = pickle.load(file)

with open('articles_list_poiniko.pkl', 'rb') as file:
    articles_list_poiniko = pickle.load(file)


with open('document_embeddings.pkl', 'rb') as file:
    document_embeddings = pickle.load(file)

with open('document_embeddings_poiniko.pkl', 'rb') as file:
    document_embeddings_poiniko = pickle.load(file)



# Function to calculate similarity
def calculate_similarity(user_input, document_embeddings_poiniko, articles_list_poiniko):



    # Tokenize and encode documents and user input
    user_input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt")

    max_sequence_length = 512

    with open('document_embeddings.pkl', 'rb') as file:
        document_embeddings = pickle.load(file)

    with open('document_embeddings_poiniko.pkl', 'rb') as file:
        document_embeddings_poiniko = pickle.load(file)

    # print(document_embeddings[0])
    # del document_embeddings[0]

    user_embedding = model(user_input_ids)[0][:, 0, :].detach().numpy()  # Detach tensor and then convert to NumPy
    # user_embedding = model(user_input_ids)[0][:, 0, :].numpy()  # Extract [CLS] token embedding for user input
    user_embedding = user_embedding.reshape(1, -1)  # Reshape user_embedding to (1, embedding_size)


    # Calculate cosine similarity between user input and documents
    similarity_scores = cosine_similarity(user_embedding, document_embeddings_poiniko)

    # print(similarity_scores)

    # Get the indices of the top three most similar documents
    top_three_indices = np.argsort(similarity_scores[0])[-10:][::-1]  # Retrieve indices of top 3 documents

    print(top_three_indices)

    # print(len(document_embeddings))
    # print(len(similarity_scores))

    # Retrieve the most similar documents

    


    print(articles_list_poiniko[359])


        # ...

    # Return top three documents and their similarity scores
    return top_three_indices, similarity_scores

# Streamlit app
def main():
    st.title('Αναζήτηση στον κώδικα...')

    user_input = st.text_input('Enter a word or phrase:', '')

    if st.button('Εύρεση σχετικών άρθρων'):
        if user_input:
            # Calculate similarity
            top_three_indices, similarity_scores = calculate_similarity(user_input, document_embeddings_poiniko, articles_list_poiniko)
            top_three_documents = [articles_list_poiniko[i] for i in top_three_indices]


            # Print the top three most similar documents and their similarity scores
            for idx, doc in zip(top_three_indices, top_three_documents):
                similarity_score = similarity_scores[0][idx]
                print(f"Similarity Score: {similarity_score:.4f} - Article: {idx} -  Document: {doc}")

                

            you = ''
            count = -1
            for i in top_three_documents:
                count+=1
                you += str(top_three_indices[count]) + ') '
                you += i
                you += "\n"


            st.write('Σχετικό άρθρο:')

            st.write(you)
        else:
            st.warning('Please enter a word or phrase.')

if __name__ == "__main__":
    main()
