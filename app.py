import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
import plotly.express as px
import openai
import os


# --- CONFIGURE OPENAI ---
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment or .streamlit/secrets.toml

# --- MODEL OPTIONS ---
MODEL_SIMPLE = "bhadresh-savani/distilbert-base-uncased-emotion"
MODEL_ADVANCED = "joeddav/distilbert-base-uncased-go-emotions-student"

# --- Load model/tokenizer by mode ---
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    labels = model.config.id2label
    return tokenizer, model, labels

# --- Get emotion predictions ---
def get_emotions(text, tokenizer, model, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].tolist()
    return sorted(
        [{"label": labels[i], "score": prob} for i, prob in enumerate(probs)],
        key=lambda x: x["score"],
        reverse=True
    )

# --- OpenAI fallback ---
def get_emotions_openai(text):
    prompt = f"Analyze the emotion in this sentence: \"{text}\". Return the top 3 emotions with a percentage value for each."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an emotion detection assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )
    return response.choices[0].message.content

# --- Streamlit UI setup ---
st.set_page_config(page_title="FeelMap – Emotion Insight", layout="centered")
st.title("🧠 FeelMap – Emotion Insight")
st.markdown("Analyze text to detect overlapping emotions using either a fast model or an AI-powered GPT model.")

# --- Sidebar model selector ---
model_choice = st.sidebar.radio("Choose detection model:", ["Simple (7 emotions)", "Advanced (28 emotions)", "OpenAI GPT (custom)"])

# --- Main input ---
text = st.text_area("Enter your text to analyze emotion:", height=200)

if text.strip():
    st.subheader("🔍 Analyzing...")

    if model_choice == "OpenAI GPT (custom)":
        try:
            result_text = get_emotions_openai(text)
            st.markdown("### 🧠 OpenAI Analysis Result")
            st.code(result_text)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
    else:
        model_name = MODEL_SIMPLE if model_choice.startswith("Simple") else MODEL_ADVANCED
        tokenizer, model, labels = load_model(model_name)
        results = get_emotions(text, tokenizer, model, labels)
        df = pd.DataFrame(results)

        st.markdown("### 🏆 Top Emotions")
        for i in range(3):
            label = results[i]['label'].capitalize()
            score = results[i]['score'] * 100
            st.markdown(f"- **{label}** ({score:.1f}%)")

        chart_df = df[df["score"] > 0.05].sort_values(by="score", ascending=False).head(10)
        chart_df["percentage"] = chart_df["score"] * 100

        st.subheader("📊 Emotion Probability Chart")

        # Sort bar order by descending probability
        chart_df = chart_df.sort_values(by="percentage", ascending=False)
        chart_df["label"] = pd.Categorical(chart_df["label"], categories=chart_df["label"], ordered=True)

        fig = px.bar(
            chart_df,
            x="label",
            y="percentage",
            text=chart_df["percentage"].apply(lambda x: f"{x:.1f}%"),
            title=None,
            labels={"label": "Emotion", "percentage": "Probability (%)"},
        )

        fig.update_traces(textposition="outside", marker_color="#1f77b4")
        fig.update_layout(
            yaxis_range=[0, max(chart_df["percentage"].max(), 50) * 1.2],
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            margin=dict(t=10)
        )
        st.plotly_chart(fig, use_container_width=True)


        with st.expander("⚠️ Was this emotion incorrect? Help improve FeelMap"):
            correct_emotion = st.text_input("What emotion would you have expected instead?")
            explanation = st.text_area("Can you explain why? (optional)")
            if st.button("Submit Feedback"):
                log_path = Path("feedback_log.csv")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f'"{text.strip()}","{correct_emotion.strip()}","{explanation.strip()}"\n')
                st.success("✅ Thank you! Your feedback has been saved.")
st.caption("Built with ❤️ using [Hugging Face](https://huggingface.co/) and [OpenAI](https://openai.com/) and [Streamlit](https://streamlit.io/)")
