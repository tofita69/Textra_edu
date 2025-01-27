from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st
from transformers import pipeline
import sys
import re
import requests
import json

# Use a free model like GPT-Neo
langchain_model_name = "google/flan-t5-base"  # Example GPT-Neo model on Hugging Face
langchain_tokenizer = AutoTokenizer.from_pretrained(langchain_model_name)
langchain_model = AutoModelForSeq2SeqLM.from_pretrained(langchain_model_name)

# Create a pipeline for summarization
langchain_pipeline = pipeline(
    "text2text-generation",
    model=langchain_model,
    tokenizer=langchain_tokenizer,
    device=langchain_model.device,  # Ensure the pipeline is on the same device as the model
    max_new_tokens=512,
)

# Ollama API interaction function
def get_ollama_response(prompt, model="tinyllama"):  # Default model is llama2
    try:
        url = "http://127.0.0.1:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "prompt": prompt, "stream": True} # stream: False for single response
        with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
            response.raise_for_status()
            full_response = "" # Accumulate the response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_line = json.loads(decoded_line)
                        if "response" in json_line:
                            full_response += json_line["response"]
                    except json.JSONDecodeError as e:
                        st.error(f"JSONDecodeError: {e}. Line: {decoded_line}") # Handle JSON errors
            return full_response
    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}"


# Streamlit App Title
st.title("Textra-edu")
st.markdown("Using Fine-tuned T5 Model for Summarization and Simple Question Answering")

# Sidebar for Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Summarizer", "Chatbot"])



# Summarizer Section
if app_mode == "Summarizer":
    st.header("Summarizer")
    text = st.text_area("Please Input a Scientific Text")

    max_length = 750
    min_length = 250

    def generate_summary(input_text):
        try:
            inputs = langchain_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding="max_length"
            )
            input_ids = inputs["input_ids"].to(langchain_model.device)
            attention_mask = inputs["attention_mask"].to(langchain_model.device)

            generated_ids = langchain_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,  # You can keep this if you're enabling sampling
                    do_sample=True  # Enable sampling
                )

            return langchain_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    if st.button("Summarize"):
        if not text.strip():
            st.error("Please fill the input field.")
        else:
            summary = generate_summary(text)
            if summary:
                st.subheader("Summary")
                st.success(summary)
                st.session_state.summary = summary

# Chatbot Section (Modified)
elif app_mode == "Chatbot":
    st.header("Chat with Textra-edu Bot")

    if "summary" not in st.session_state:
        st.write("Please summarize a text first.")
    else:
        summary = st.session_state.summary
        st.write(f"Chatting based on the following summary:\n\n{summary}")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the summary."}]

        chat_placeholder = st.container()

        with chat_placeholder:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.write(f"**You:** {msg['content']}")
                else:
                    st.write(f"**Bot:** {msg['content']}")

        user_input = st.text_input("Your Message", placeholder="Type your question here...")

        if st.button("Send"):
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})

                try:
                    # Use Ollama for generating the response based on the summary and user input
                    prompt = f"Context: {summary}\nUser Question: {user_input}\nAnswer:"
                    bot_response = get_ollama_response(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "I encountered an error processing your request."})

                st.rerun()
