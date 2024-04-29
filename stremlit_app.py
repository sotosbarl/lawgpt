import requests
from transformers import  pipeline
import time

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}



classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

with open('my_dict.pickle', 'rb') as file:
    dictionary = pickle.load(file)

def classify(text,labels):
    output = classifier(text, labels, multi_label=False)
    
    return output


def query(payload, retries=3, wait_time=5):
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


text = st.text_input('Enter some text:')  # Input field for new text

if text:

    labels = list(dictionary)
    
    output = classify(text,labels)

    output = output["labels"][0]

    labels = list(dictionary[output])

    output2 = classify(text,labels)

    output2 = output2["labels"][0]


    answer = dictionary[output][output2]

    # Create a translator object with specified source and target languages
    translator = Translator(from_lang='el', to_lang='en')
    translator2 = Translator(from_lang='en', to_lang='el')

 

# Translate the text from Greek to English
    answer = translator.translate(answer)
    text = translator.translate(text)


    
output = query({
  "inputs": "Based on this info only:" + answer +" ,answer this question, by reasoning step by step:" + text,
})

if output:

    translated_text2 = translator2.translate(out[0]['generated_text'])
    st.text(output)

    st.text(translated_text2)

else:
    print("Failed to get response from the model.")
