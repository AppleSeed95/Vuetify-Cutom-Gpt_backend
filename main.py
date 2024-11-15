from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
from openai import OpenAI
import tiktoken
import sqlite3
import numpy as np
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

origins = [
    "http://localhost:3000",   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,               
    allow_credentials=True,              
    allow_methods=["*"],                  
    allow_headers=["*"],                  
)

client = OpenAI(api_key=api_key)

DB_NAME = "embeddings.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_embedding(filename: str, content: str, embedding: list):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO embeddings (filename, content, embedding) VALUES (?, ?, ?)
    ''', (filename, content, sqlite3.Binary(np.array(embedding, dtype=np.float32).tobytes())))
    conn.commit()
    conn.close()

def num_tokens_from_string(string):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text, max_tokens=200):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        temp_chunk = current_chunk + [word]
        temp_text = " ".join(temp_chunk)
        new_token_count = num_tokens_from_string(temp_text)
        
        if new_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def search_most_similar_embedding(user_embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT content, embedding FROM embeddings')
    results = cursor.fetchall()
    conn.close()

    user_embedding = np.array(user_embedding)
    similarities = []

    for content, embedding_blob in results:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(user_embedding, embedding)
        similarities.append((content, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_contents = [content for content, sim in similarities[:2]]

    return top_similar_contents


class UserData(BaseModel):
    user_message: str

@app.post("/chat/")
async def chat(user_data: UserData):
    response = client.embeddings.create(
        input=user_data.user_message,
        model="text-embedding-ada-002"
    )
    user_embedding = response.data[0].embedding

    most_similar_content = search_most_similar_embedding(user_embedding)
    if most_similar_content:
        async def event_stream():
            stream = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Based on the following content:\n{most_similar_content}\n\nUser: {user_data.user_message}\n\nAI:"}],
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        raise HTTPException(status_code=404, detail="No similar content found in embeddings.")

@app.post("/test")
async def test_endpoint():
    return {"message": "hello"}

st.title("AI Model Interaction Platform")

uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        content = df.to_string()
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        content = df.to_string()
    elif uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
        content = None

    if content:
        chunks = chunk_text(content, max_tokens=500)
        for chunk in chunks:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding

            save_embedding(uploaded_file.name, chunk, embedding)

        st.success("File uploaded and embeddings saved successfully.")


        # if st.button("Send"):
#     if user_message:
#         # Generate embedding for the user's message
#         response = client.embeddings.create(
#             input=user_message,
#             model="text-embedding-ada-002"
#         )
#         user_embedding = response.data[0].embedding

#         # Search for the most similar embedding in the database
#         most_similar_content = search_most_similar_embedding(user_embedding)

#         response_placeholder = st.empty()

#         if most_similar_content:
#             # Stream response from OpenAI API
#             stream = client.chat.completions.create(
#                 model="gpt-4",
#                 messages=[{"role": "user", "content": f"Based on the following content:\n{most_similar_content}\n\nUser: {user_message}\n\nAI:"}],
#                 stream=True
#             )

#             # Stream the response content
#             response_text = ""
#             for chunk in stream:
                
#                 if chunk.choices[0].delta.content is not None:
#                     response_text += chunk.choices[0].delta.content
#                     response_placeholder.write(response_text, unsafe_allow_html=True)
                    
#         else:
#             st.error("No similar content found in embeddings.")
