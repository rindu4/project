import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Model Training & Caching ---

@st.cache_resource
def train_model():
    """
    Trains and caches the TF-IDF Vectorizer and Logistic Regression Model.
    
    NOTE: Since external data persistence is not guaranteed, a small,
    hardcoded internal dataset is used for demonstration purposes.
    """
    st.info("Training model on internal dataset...")

    # Minimal, hardcoded dataset for demonstration
    data = {
        'headline': [
            "Trump meets with Putin to discuss trade deal",
            "Stock market hits record high after Fed announcement",
            "Scientists discover new Earth-like planet nearby",
            "Local government passes sweeping climate change law",
            "Queen Elizabeth endorses Donald Trump for President in shock move",
            "Aliens landed in Texas yesterday, White House confirms cover-up",
            "Sun will explode next week, experts warn world is ending",
            "Doctor recommends drinking coffee for immediate cancer cure",
        ],
        'label': [1, 1, 1, 1, 0, 0, 0, 0] # 1: Real, 0: Fake
    }
    df = pd.DataFrame(data)

    X = df['headline']
    y = df['label']

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X)

    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_tfidf, y)
    
    st.success("Model training complete! Ready for predictions.")
    return model, vectorizer

# Load the trained components
model, vectorizer = train_model()

# --- 2. Prediction Functions ---

def predict_headline(headline, model, vectorizer):
    """Predicts label and confidence for a single headline."""
    if not headline:
        return None, None

    # Vectorize the input headline
    headline_vector = vectorizer.transform([headline.lower()])
    
    # Predict label (0 or 1)
    prediction = model.predict(headline_vector)[0]
    
    # Predict probabilities (for confidence)
    probabilities = model.predict_proba(headline_vector)[0]
    
    # Get confidence for the predicted class
    confidence = probabilities[prediction]
    
    return prediction, confidence

def batch_predict(uploaded_file, model, vectorizer):
    """Handles CSV upload and returns a DataFrame with predictions."""
    if uploaded_file is None:
        return None

    try:
        df = pd.read_csv(uploaded_file)
        
        # Assume the headline column is the first text column or named 'headline'
        if 'headline' in df.columns:
            text_column = 'headline'
        elif len(df.columns) > 0 and df.dtypes.iloc[0] == 'object':
            text_column = df.columns[0]
        else:
            st.error("Could not find a suitable text column (expected 'headline' or a column with text data).")
            return None

        # Prepare headlines for prediction
        headlines = df[text_column].fillna('').astype(str).str.lower()
        
        # Vectorize all headlines
        headlines_vector = vectorizer.transform(headlines)
        
        # Predict
        predictions = model.predict(headlines_vector)
        probabilities = model.predict_proba(headlines_vector)
        
        # Get confidence for the predicted class
        confidences = [prob[pred] for pred, prob in zip(predictions, probabilities)]

        # Add results to the DataFrame
        df['Predicted_Label'] = predictions
        df['Prediction'] = np.where(predictions == 1, 'REAL', 'FAKE')
        df['Confidence'] = [f"{c*100:.2f}%" for c in confidences]

        return df, text_column

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        st.stop()


# --- 3. Streamlit UI Layout ---

st.title("📰 Real-Time Fake News Detector")
st.markdown("Use this tool to check the authenticity of news headlines using a trained Logistic Regression model leveraging TF-IDF.")

# --- Single Headline Prediction Section ---
st.header("1. Check a Single Headline")

input_headline = st.text_area(
    "Enter the news headline below:", 
    placeholder="Example: Global stock markets surge after major tech breakthrough...",
    height=100
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze Headline", type="primary", use_container_width=True):
        if input_headline:
            with st.spinner('Analyzing...'):
                prediction, confidence = predict_headline(input_headline, model, vectorizer)
            
            if prediction is not None:
                confidence_percent = confidence * 100
                
                if prediction == 1:
                    result_text = "REAL"
                    icon = "✅"
                    color = "green"
                    st.success(f"{icon} Prediction: {result_text}")
                else:
                    result_text = "FAKE"
                    icon = "❌"
                    color = "red"
                    st.error(f"{icon} Prediction: {result_text}")

                # Display confidence in a nice card
                st.markdown(
                    f"""
                    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid {color}; margin-top: 15px;'>
                        <p style='margin: 0; font-size: 1.1em;'>Confidence: <strong>{confidence_percent:.2f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter a headline to analyze.")

# --- Batch Prediction Section ---
st.header("2. Batch Check (CSV Upload)")
st.caption("Upload a CSV file where one column contains news headlines.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    accept_multiple_files=False,
    help="The first column of text data will be treated as the headlines."
)

if uploaded_file is not None:
    st.info(f"File '{uploaded_file.name}' uploaded successfully.")
    
    if st.button("Process Batch Predictions", use_container_width=True):
        with st.spinner('Processing all headlines...'):
            results_df, text_column = batch_predict(uploaded_file, model, vectorizer)
            
            if results_df is not None:
                st.subheader(f"Batch Results (Classified based on column: '{text_column}')")
                
                # Show results in an interactive table
                st.dataframe(results_df[[text_column, 'Prediction', 'Confidence', 'Predicted_Label']], use_container_width=True)
                
                # Prepare CSV for download
                csv_output = results_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv_output,
                    file_name='classified_headlines.csv',
                    mime='text/csv',
                    help="Click to download the original data plus the prediction columns."
                )
                
                # Simple summary
                fake_count = (results_df['Predicted_Label'] == 0).sum()
                real_count = (results_df['Predicted_Label'] == 1).sum()
                
                st.markdown(f"""
                ---
                **Summary:**
                - Total Headlines Processed: **{len(results_df)}**
                - Predicted as REAL: **{real_count}**
                - Predicted as FAKE: **{fake_count}**
                """)
