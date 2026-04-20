import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(page_title="Next Word Predictor", page_icon="🚀")

# 1. Load the pre-trained components
@st.cache_resource
def load_assets():
    # Load the model
    # If the error persists after upgrade, we load without compilation
    model = load_model('lstm_model1.h5', compile=False)
    
    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    # Load max_len (Value is 745 based on your data)
    with open('max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
        
    return model, tokenizer, max_len

try:
    model, tokenizer, max_len = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Try running: pip install --upgrade tensorflow")
    st.stop()

# 2. Prediction logic
def predict_next_words(seed_text, next_words_count):
    words_predicted = []
    current_text = seed_text
    
    for _ in range(next_words_count):
        # Preprocessing
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
        
        # Prediction
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Convert index to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        if not output_word:
            break
            
        current_text += " " + output_word
        words_predicted.append(output_word)
        
    return current_text

# 3. Streamlit UI
st.title("Next Word Predictor 🚀")
st.markdown("---")

user_input = st.text_input("Enter your text:", placeholder="e.g., what is the meaning of")
num_words = st.slider("Words to generate:", 1, 20, 5)

if st.button("Predict Next Words"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = predict_next_words(user_input, num_words)
            st.subheader("Prediction:")
            st.success(result)
    else:
        st.warning("Please enter some starting text.")

# Information about the model
with st.expander("Model Details"):
    st.write(f"**Vocabulary Size:** {len(tokenizer.word_index)}")
    st.write(f"**Max Sequence Length:** {max_len}")
    st.write("Built with LSTM and trained on a quotes dataset.")