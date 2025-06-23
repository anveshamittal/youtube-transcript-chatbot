import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def extract_video_id(input_str):
    if re.match(r"^[\w-]{11}$", input_str):
        return input_str
    patterns = [
        r"youtu\.be/([^\?&/]+)",
        r"youtube\.com/watch\?v=([^\?&/]+)",
        r"youtube\.com/embed/([^\?&/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return match.group(1)
    return None

with st.sidebar:
    st.header("🔑 Configuration")
    provider = st.selectbox("Select provider", ["OpenAI", "Groq"])
    api_key = st.text_input("Enter your API key", type="password")
    base_url = st.text_input("Enter base URL (optional for OpenAI/Azure)")
    video_input = st.text_input("Enter YouTube video URL or ID")

if not api_key or not video_input:
    st.warning("Please enter required fields in the sidebar.")
    st.stop()

video_id = extract_video_id(video_input)
if not video_id:
    st.error("Could not extract a valid YouTube video ID from your input.")
    st.stop()

if provider == "OpenAI":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        model="gpt-4o-mini-custom",
        default_query={"api-version": "preview"} if base_url else None
    )
elif provider == "Groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama3-8b-8192"
    )
else:
    st.error("Invalid provider selected.")
    st.stop()

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    st.error("No captions available for this video.")
    st.stop()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

if "history" not in st.session_state:
    st.session_state.history = []

def build_prompt(context, history, question):
    history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
    return f"""
You are a helpful assistant. 
Answer ONLY from the provided transcript context in not more than 50 words.
If the context is insufficient, just say you don't know.

Transcript context:
{context}

Chat history:
{history_text}

Question: {question}
"""

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

st.title("🎥 YouTube Transcript Chatbot")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

user_question = st.chat_input("Ask a question about the video")
if user_question:
    result = parallel_chain.invoke(user_question)
    prompt_text = build_prompt(result['context'], st.session_state.history, user_question)
    output = llm.invoke(prompt_text)
    parsed = parser.invoke(output)
    
    with st.chat_message("user"):
        st.write(user_question)
    with st.chat_message("assistant"):
        st.write(parsed)

    st.session_state.history.append((user_question, parsed))
