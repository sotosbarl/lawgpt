from transformers import BertModel, BertTokenizer , pipeline, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import docx
import torch
import streamlit as st




# tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)


# gpt_tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
gpt_model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

import requests

import docx 
import pickle
import os


# session = requests.Session()
# session.headers = SESSION_HEADERS
# session.cookies.set("__Secure-1PSID", token)
# session.cookies.set("__Secure-1PSIDTS","xxxxx")
# session.cookies.set("__Secure-1PSIDCC","xxxxx")

# bard = Bard(token=token, session=session)


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


            # model_name = 'bert-base-multilingual-cased'  # Multilingual BERT model
            # task = 'text2text-generation'
            # pipe = pipeline(task, model=model_name)


            you+= "ποια από τις παραπάνω αριθμημενες προτάσεις  μιλάει για " + user_input + ", απάντησε συγκεκριμενα, γράψε μου μόνο τη σωστη πρόταση"
            # you += "Τι λέει ο αστικος κωδικας για αυτο; χρησιμοποιησε τις παραπανω αριθμημενες προτασεις."
            # print(bard.get_answer(you)['content'])
                # Your similarity calculation code
            # text = pipe("once upon a time...")
            st.write('Σχετικό άρθρο:')
            # for i in range(3):
                
            #     st.write( f"Άρθρο: {top_three_indices[i]} {top_three_documents[i]}")

            # text = bard.get_answer(you)['content']
            pipe = pipeline("text2text-generation", model=gpt_model)
            x = pipeline("Three movies of Morgan Freeman")
            st.write(x)
        else:
            st.warning('Please enter a word or phrase.')

if __name__ == "__main__":
    main()
