import requests
# from transformers import  pipeline
import time
import pickle
import streamlit as st
from translate import Translator

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}

API_URL2 = "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

with open('my_dict.pickle', 'rb') as file:
    dictionary = pickle.load(file)

def classify(text,labels):
    output = classifier(text, labels, multi_label=False)
    
    return output


def query(payload, API_URL, retries=3, wait_time=5):
  """
  This function sends a query to the Hugging Face Inference API with retry logic.
  """
  for attempt in range(retries):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
      return response.json()
    elif response.status_code == 429:  # Handle rate limiting errors
      print(f"Rate limit reached, retrying in {wait_time} seconds...")
      time.sleep(wait_time)
    else:
      # Handle other errors (e.g., model loading error)
      print(f"Error: {response.json()}")
      return None
  print(f"Failed to query the model after {retries} attempts.")
  return None


#CHAT HERE


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ρωτήστε κάτι:"):
  # Display user message in chat message container
  st.chat_message("user").markdown(prompt)
  # Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

    

# text = st.text_input('Ρωτήστε κάτι:')  # Input field for new text

# if text:
  text = prompt

  labels = list(dictionary)

  output = query({
    "inputs": text,
    "parameters": {"candidate_labels": labels},
},API_URL2)



  # output = classify(text,labels)

  output = output["labels"][0]

  labels = list(dictionary[output])

  output2 = query({
    "inputs": text,
    "parameters": {"candidate_labels": labels},
},API_URL2)

  output2 = output2["labels"][0]


  # output2 = classify(text,labels)

  # output2 = output2["labels"][0]


  answer = dictionary[output][output2]


  # Create a translator object with specified source and target languages
  translator = Translator(from_lang='el', to_lang='en')
  translator2 = Translator(from_lang='en', to_lang='el')


  answer_rest = answer
  answer_translate = ''
  while len(answer_rest)>500:
    answer_begin = answer_rest[0:500]
    answer_rest = answer_rest[500:-1]
  # Translate the text from Greek to English
  answer_begin = translator.translate(answer_begin)
  answer_translate +=answer_begin

  text = translator.translate(text)
  final_prompt = "Based on this info only: " + answer_translate +" ,answer this question:" + text


  output = query({
  "inputs": final_prompt,
  },API_URL)

  if output:


    answer_gpt = output[0]['generated_text']

    split_text = answer_gpt.split(final_prompt)
    answer_gpt = split_text[1]

    answer_rest = answer_gpt
    answer_translate = ''
    while len(answer_rest)>500:
      answer_begin = answer_rest[0:500]
      answer_rest = answer_rest[500:-1]

      answer_begin = translator2.translate(answer_begin)
      answer_translate +=answer_begin


    answer_rest = translator2.translate(answer_rest)
    answer_translate +=answer_rest

    # st.text(answer_translate)

    response = answer_translate
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


  else:
    print("Failed to get response from the model.")
