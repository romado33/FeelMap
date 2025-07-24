# app.py
import streamlit as st
import pandas as pd
from feelmap_model import detect_emotions

st.set_page_config(page_title="FeelMap", page_icon="🧠")
st.title("🧭 FeelMap: Emotion Navigator for Text")
st.write("Designed with clarity and support for autistic users in mind.")

user_input = st.text_area("Enter a sentence or message to analyze:", height=150)

if st.button("Analyze Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        results = detect_emotions(user_input)
        df = pd.DataFrame(results)

        st.markdown("### 🎯 Most Likely Emotion")
        top = results[0]
        st.success(f"**{top['label'].capitalize()}** ({top['score']*100:.1f}%)")

        st.markdown("### 📊 Emotion Probability Chart")
        st.bar_chart(df.set_index("label"))

        st.markdown("### 🧠 Feedback")
        rating = st.radio("Did this feel accurate?", ["👍 Yes", "🤷 Not sure", "👎 No"])
        if rating == "👎 No":
            feedback = st.text_area("Tell us how *you* understood the emotion:")
            if st.button("Submit Feedback"):
                with open("feedback.txt", "a") as f:
                    f.write(f"Input: {user_input}\nFeedback: {feedback}\n---\n")
                st.success("Thank you! Your feedback helps improve FeelMap.")
