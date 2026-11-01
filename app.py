import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Cricket AI", page_icon="🏏", layout="centered")

st.title("🏏 AI Cricket Prediction")
st.write("Yeh AI 9,000+ matches ke data par train hua hai. Teams select karein aur prediction dekhein!")

# Model load karna
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

model, encoders = load_assets()

# Dropdowns ke liye list nikalna
teams = list(encoders['team1'].classes_)
formats = list(encoders['format'].classes_)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams)
    toss_winner = st.selectbox("Toss Winner", teams)
with col2:
    team2 = st.selectbox("Team 2", teams)
    toss_decision = st.selectbox("Toss Decision", list(encoders['toss_decision'].classes_))

match_format = st.selectbox("Match Format", formats)
venue = st.selectbox("Venue", list(encoders['venue'].classes_))
city = st.selectbox("City", list(encoders['city'].classes_))

if st.button("Predict Winner! 🚀"):
    try:
        # User input ko encode karna
        input_data = pd.DataFrame({
            'format': [encoders['format'].transform([match_format])[0]],
            'venue': [encoders['venue'].transform([venue])[0]],
            'city': [encoders['city'].transform([city])[0]],
            'team1': [encoders['team1'].transform([team1])[0]],
            'team2': [encoders['team2'].transform([team2])[0]],
            'toss_winner': [encoders['toss_winner'].transform([toss_winner])[0]],
            'toss_decision': [encoders['toss_decision'].transform([toss_decision])[0]]
        })

        # Probability nikalna
        prob = model.predict_proba(input_data)[0]
        
        team1_prob = prob[1] * 100
        team2_prob = prob[0] * 100

        st.success("Prediction Complete!")
        if team1_prob > team2_prob:
            st.markdown(f"### 🏆 Winner: **{team1}**")
        else:
            st.markdown(f"### 🏆 Winner: **{team2}**")
            
        st.write(f"📊 **{team1}** Win Probability: **{team1_prob:.1f}%**")
        st.write(f"📊 **{team2}** Win Probability: **{team2_prob:.1f}%**")
        
    except Exception as e:
        st.error(f"Error: {e}")
print("✅ app.py file ban gayi!")
