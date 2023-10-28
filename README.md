---
title: Ai Chatbot with Conversational Memory
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.28.0
app_file: main.py
pinned: false
license: mit
---

# Streamlit + Langchain + LLama.cpp w/ Mistral + Conversational Memory

Run your own AI Chatbot locally without a GPU.

To make that possible, we use the [Mistral 7b](https://mistral.ai/news/announcing-mistral-7b/) model.  
However, you can use any quantized model that is supported by [llama.cpp](https://github.com/ggerganov/llama.cpp).

This model will chatbot will allow you to define it's personality and respond to the questions accordingly.  
This example remembers the chat history allowing you to ask follow up questions.

# TL;DR instructions

1. Install llama-cpp-python
2. Install langchain
3. Install streamlit
4. Run streamlit

# Step by Step instructions

The setup assumes you have `python` already installed and `venv` module available.

1. Download the code or clone the repository.
2. Inside the root folder of the repository, initialize a python virtual environment:
```bash
python -m venv venv
```
3. Activate the python envitonment:
```bash
source venv/bin/activate
```
4. Install required packages (`langchain`, `llama.cpp`, and `streamlit`):
```bash
pip install -r requirements.txt
```
5. Start `streamlit`:
```bash
streamlit run main.py
```
6. The `models` directory will be created and the app will download the `Mistral7b` quantized model from `huggingface` from the following link:
[mistral-7b-instruct-v0.1.Q4_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf)

# Screenshot

![Screenshot from 2023-10-23 20-36-22](https://github.com/lalanikarim/ai-chatbot-conversational/assets/1296705/9919a3b3-c24e-4b1d-a121-1eb6724db5c6)
