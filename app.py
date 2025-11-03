import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# -------------------------------
# Load model and tokenizer
# -------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("sentiment_lstm_model.h5")
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

MAX_LEN = 200

# -------------------------------
# Page Config & Custom CSS
# -------------------------------
st.set_page_config(page_title="🎭 Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Animated gradient background */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Hero Section */
        .hero-section {
            text-align: center;
            padding: 40px 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 30px;
            margin: 20px auto 40px;
            max-width: 900px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .hero-title {
            font-size: 56px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            animation: fadeInDown 0.8s ease-out;
        }
        
        .hero-subtitle {
            font-size: 20px;
            color: #e0e0e0;
            font-weight: 300;
            animation: fadeInUp 0.8s ease-out;
        }
        
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 15px;
            font-size: 14px;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Input Card */
        .input-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        
        .input-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        }
        
        .section-title {
            font-size: 24px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Text Area Styling */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.15) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            color: white !important;
            font-size: 16px !important;
            padding: 15px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextArea textarea:focus {
            border: 2px solid rgba(255, 255, 255, 0.5) !important;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.2) !important;
        }
        
        .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.5) !important;
        }
        
        /* Example Cards */
        .example-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px 20px;
            border-radius: 15px;
            margin: 10px 0;
            border-left: 4px solid;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .example-card:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateX(5px);
        }
        
        .example-positive {
            border-color: #10b981;
        }
        
        .example-negative {
            border-color: #ef4444;
        }
        
        /* Button Styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 15px 40px !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            border-radius: 50px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Result Card */
        .result-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 40px;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: slideIn 0.5s ease-out;
            margin-top: 30px;
        }
        
        .sentiment-icon {
            font-size: 80px;
            animation: bounce 1s ease;
        }
        
        .sentiment-label {
            font-size: 36px;
            font-weight: 700;
            margin: 20px 0;
        }
        
        .positive-sentiment {
            color: #10b981;
            text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        }
        
        .negative-sentiment {
            color: #ef4444;
            text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        }
        
        .confidence-container {
            margin: 30px auto;
            max-width: 400px;
        }
        
        .confidence-label {
            font-size: 18px;
            color: #e0e0e0;
            margin-bottom: 10px;
        }
        
        .confidence-value {
            font-size: 32px;
            font-weight: 700;
            color: #ffffff;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            height: 12px;
        }
        
        /* Stats Cards */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
        }
        
        .stat-label {
            font-size: 14px;
            color: #d0d0d0;
            margin-top: 5px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 30px;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 50px;
        }
        
        .footer-link {
            color: #ffffff;
            text-decoration: none;
            font-weight: 600;
        }
        
        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }
        
        /* Warning and Info boxes */
        .stAlert {
            border-radius: 15px !important;
            backdrop-filter: blur(10px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Hero Section
# -------------------------------
st.markdown("""
    <div class='hero-section'>
        <div class='hero-title'>🎭 AI Sentiment Analyzer</div>
        <div class='hero-subtitle'>Unlock emotions hidden in text with advanced LSTM Neural Networks</div>
        <div class='hero-badge'>⚡ Powered by Deep Learning</div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# Main Layout
# -------------------------------
col1, col2 = st.columns([1.3, 1], gap="large")

with col1:
    st.markdown("""
        <div class='input-card'>
            <div class='section-title'>✍️ Share Your Thoughts</div>
        </div>
    """, unsafe_allow_html=True)
    
    user_review = st.text_area(
        "",
        height=250,
        placeholder="✨ Type or paste your movie review here... Let AI decode the sentiment!",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.markdown("""
        <div class='input-card'>
            <div class='section-title'>💡 Try These Examples</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='example-card example-positive'>
            <strong>😊 Positive Example</strong><br>
            <em>"I absolutely loved this movie! The cinematography was breathtaking and the acting was phenomenal."</em>
        </div>
        
        <div class='example-card example-negative'>
            <strong>😞 Negative Example</strong><br>
            <em>"What a waste of time. The plot was boring and the characters were poorly developed."</em>
        </div>
        
        <div class='example-card example-positive'>
            <strong>😊 Positive Example</strong><br>
            <em>"Brilliant storytelling! This film kept me on the edge of my seat from start to finish."</em>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
if analyze_button:
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review to analyze!")
    else:
        with st.spinner("🔄 Analyzing sentiment..."):
            sequence = tokenizer.texts_to_sequences([user_review])
            padded = pad_sequences(sequence, maxlen=MAX_LEN)
            prediction = model.predict(padded)[0][0]

            is_positive = prediction > 0.5
            confidence = prediction if is_positive else 1 - prediction

        # Result Display
        st.markdown("<br>", unsafe_allow_html=True)
        
        if is_positive:
            icon = "😊"
            sentiment_text = "POSITIVE"
            sentiment_class = "positive-sentiment"
        else:
            icon = "😞"
            sentiment_text = "NEGATIVE"
            sentiment_class = "negative-sentiment"

        st.markdown(f"""
            <div class='result-card'>
                <div class='sentiment-icon'>{icon}</div>
                <div class='sentiment-label {sentiment_class}'>{sentiment_text}</div>
                
                <div class='confidence-container'>
                    <div class='confidence-label'>Confidence Level</div>
                    <div class='confidence-value'>{confidence*100:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.progress(float(confidence))
        
        # Additional Stats
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{len(user_review.split())}</div>
                    <div class='stat-label'>Words Analyzed</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{len(user_review)}</div>
                    <div class='stat-label'>Characters</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-value'>{prediction:.3f}</div>
                    <div class='stat-label'>Raw Score</div>
                </div>
            """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
    <div class='footer'>
        <p>Crafted with ❤️ by <a href='#' class='footer-link'>Daniyal Aslam</a></p>
        <p>🚀 Leveraging LSTM Neural Networks for Sentiment Analysis</p>
    </div>
""", unsafe_allow_html=True)