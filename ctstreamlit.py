import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import re

st.set_page_config(
    page_title="Comment Toxicity Detector",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)


TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 150

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_artifacts():
    try:
        model = load_model('toxicity_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

st.markdown("<div class='main-header'>Comment Toxicity Detector</div>", unsafe_allow_html=True)

model, tokenizer, err = load_artifacts()

if err:
    st.warning("⚠️ Pre-trained model weights (`toxicity_model.h5`) or tokenizer (`tokenizer.pkl`) were not found. Running in simulation mode for UI testing.")


st.header("Comment Analyzer")
st.write("Enter a comment below to identify the toxicity probability across all categories")

user_input = st.text_area(
    "Enter Comment:",
    height=120,
    placeholder="Type or paste a comment here..."
)

if st.button("Analyze Comment", type="primary"):
    if not user_input.strip():
        st.write("Please enter text before running analysis.")
    else:
        cleaned = clean_text(user_input)
        
        if model and tokenizer:
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            preds = model.predict(padded)[0]

        threshold = 0.5
        results_df = pd.DataFrame({
            "Category": [c.replace('_', ' ').title() for c in TARGET_COLS],
            "Probability Score": preds,
            
        })

        flagged_mask = preds >= threshold
        
        if any(flagged_mask):
            flagged_categories = [results_df.loc[i, "Category"] for i, val in enumerate(flagged_mask) if val]
            st.warning(f"⚠️ **Toxic Comment**")
        else:
            st.success("✅ **Clean comment**")
        
        st.markdown("Toxicity Probability Scores")
        st.dataframe(
            results_df.style.format({"Probability Score": "{:.2%}"}),
            use_container_width=True
        )

st.divider()

st.header("Bulk CSV Prediction")
st.write("Upload a CSV file containing comments to get toxicity predictions for every record.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write(f"**Loaded {len(batch_df)} rows.**")
    st.dataframe(batch_df.head())
    
    
    if st.button("Process Batch Predictions", type="primary"):
        with st.spinner("Processing batch predictions..."):
            cleaned_list = [clean_text(str(t)) for t in batch_df['comment_text']]
            
            if model and tokenizer:
                seqs = tokenizer.texts_to_sequences(cleaned_list)
                padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
                preds = model.predict(padded)
            else:
                preds = np.random.uniform(0.01, 0.25, size=(len(batch_df), len(TARGET_COLS)))
            
            for i, col in enumerate(TARGET_COLS):
                batch_df[f"{col}"] = preds[:, i]
            
            st.success("Batch Prediction Complete!")
            st.markdown("Results Overview")
            st.dataframe(batch_df)
