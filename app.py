###  using only some set of data ###
# import os
# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
# import streamlit as st
# import pandas as pd
# from openai import OpenAI
# from datetime import datetime, timedelta

# import logging
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document

# # Environment setup
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Logging setup
# log_path = "web-chat_session-2.log"
# logging.basicConfig(filename=log_path, filemode="a", level=logging.INFO)

# # Streamlit setup
# st.set_page_config(page_title="PR Insights Chatbot", layout="wide")
# st.title("PR Insights Dashboard")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# uploaded_file = st.sidebar.file_uploader("Upload Pull Request CSV", type="csv")


# @st.cache_resource(show_spinner="Building vector store...")
# def build_vector_store(df: pd.DataFrame):
#     rows = df.fillna("").astype(str).apply(lambda r: " | ".join(r), axis=1).tolist()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     documents = [Document(page_content=chunk) for row in rows for chunk in splitter.split_text(row)]
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_documents(documents, embedding_model)


# @st.cache_data(show_spinner="Generating insights...")
# def generate_insights(df: pd.DataFrame, _client: OpenAI):
#     df = df.copy()
#     df["Created_Date"] = pd.to_datetime(df["Created_Date"], errors="coerce")
#     df["Merged_Date"] = pd.to_datetime(df["Merged_Date"], errors="coerce")
#     df = df[df["Merged_Date"].notnull()]
#     df["Merge_Time_Days"] = (df["Merged_Date"] - df["Created_Date"]).dt.days

#     last_30_days = datetime.now() - timedelta(days=30)
#     recent_prs = df[df["Created_Date"] >= last_30_days]

#     avg_merge_time = recent_prs["Merge_Time_Days"].mean()
#     bottlenecks = recent_prs[recent_prs["Merge_Time_Days"] > avg_merge_time]
#     df["Week"] = df["Merged_Date"].dt.to_period("W").astype(str)
#     weekly_throughput = df.groupby("Week").size().tail(4).to_dict()

#     prompt = f"""You are a senior DevOps analyst.

# Please analyze the following PR metrics:

# 1. Average merge time in the last 30 days: {avg_merge_time:.2f} days
# 2. Bottlenecks (PRs that took longer than average): {len(bottlenecks)}
# 3. Weekly PR merge trend: {weekly_throughput}

# Based on these stats, provide:
# - Key observations
# - Possible reasons for bottlenecks
# - Suggestions for improvement in PR throughput
# - Summary of the trend (is it improving or degrading?)
# """

#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[
#             {"role": "system", "content": "You are a DevOps productivity analyst."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#         max_tokens=1024
#     )
#     return response.choices[0].message.content.strip()


# def query_chatbot(question: str, vector_store, client: OpenAI) -> str:
#     docs = vector_store.similarity_search(question, k=5)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     prompt = f"""You are an assistant analyzing pull request data.
# Use the data chunks below to answer the question.

# {context}

# Question: {question}
# """
#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[
#             {"role": "system", "content": "You are a PR insights expert. Keep responses concise and structured."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#         max_tokens=2048
#     )
#     answer = response.choices[0].message.content.strip()

#     # Save interaction to log file
#     with open(log_path, "a", encoding="utf-8") as logf:
#         logf.write(f"\n---\nUser: {question}\nResponse: {answer}\n")

#     return answer


# # Main application logic
# if uploaded_file and openai_api_key:
#     df = pd.read_csv(uploaded_file)
#     client = OpenAI(api_key=openai_api_key)
#     vector_store = build_vector_store(df)
#     insights = generate_insights(df, client)

#     st.subheader("Auto-Generated Insights")
#     st.text_area("Insight Summary", insights, height=300)

#     st.subheader("Ask a Question")
#     user_q = st.text_input("Your Question")
#     if user_q:
#         result = query_chatbot(user_q, vector_store, client)
#         st.markdown("**Response:**")
#         st.write(result)

# else:
#     st.warning("Please upload a CSV file and provide a valid OpenAI API key.")







### using entire data ###
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Logging setup
log_path = "chat_session-1.log"
logging.basicConfig(filename=log_path, filemode="a", level=logging.INFO)

# Streamlit UI
st.set_page_config(page_title="PR Insights Chatbot", layout="wide")
st.title("PR Insights Dashboard")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Pull Request CSV", type="csv")


@st.cache_resource(show_spinner="Building vector store...")
def build_vector_store(df: pd.DataFrame):
    rows = df.fillna("").astype(str).apply(lambda r: " | ".join(r), axis=1).tolist()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = [Document(page_content=chunk) for row in rows for chunk in splitter.split_text(row)]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embedding_model), documents


@st.cache_data(show_spinner="Generating insights...")
def generate_insights(df: pd.DataFrame, _client: OpenAI):
    df = df.copy()
    df["Created_Date"] = pd.to_datetime(df["Created_Date"], errors="coerce")
    df["Merged_Date"] = pd.to_datetime(df["Merged_Date"], errors="coerce")
    df = df[df["Merged_Date"].notnull()]
    df["Merge_Time_Days"] = (df["Merged_Date"] - df["Created_Date"]).dt.days

    last_30_days = datetime.now() - timedelta(days=30)
    recent_prs = df[df["Created_Date"] >= last_30_days]

    avg_merge_time = recent_prs["Merge_Time_Days"].mean()
    bottlenecks = recent_prs[recent_prs["Merge_Time_Days"] > avg_merge_time]
    df["Week"] = df["Merged_Date"].dt.to_period("W").astype(str)
    weekly_throughput = df.groupby("Week").size().tail(4).to_dict()

    prompt = f"""You are a senior DevOps analyst.

Please analyze the following PR metrics:

1. Average merge time in the last 30 days: {avg_merge_time:.2f} days
2. Bottlenecks (PRs that took longer than average): {len(bottlenecks)}
3. Weekly PR merge trend: {weekly_throughput}

Based on these stats, provide:
- Key observations
- Possible reasons for bottlenecks
- Suggestions for improvement in PR throughput
- Summary of the trend (is it improving or degrading?)
"""

    response = _client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a DevOps productivity analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()


def query_chatbot_full_context(question: str, documents, client: OpenAI) -> str:
    # Concatenate all document chunks safely within token limit
    context = "\n\n".join([doc.page_content for doc in documents])[:12000]  # limit to approx 4000 tokens

    prompt = f"""You are an assistant analyzing pull request data.
Use the complete PR dataset below to answer the user's question.

{context}

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a PR insights expert. Keep responses concise and structured."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    answer = response.choices[0].message.content.strip()

    # Log interaction
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n---\nUser: {question}\nResponse: {answer}\n")

    return answer


# Main app logic
if uploaded_file and openai_api_key:
    df = pd.read_csv(uploaded_file)
    client = OpenAI(api_key=openai_api_key)
    vector_store, docs = build_vector_store(df)
    insights = generate_insights(df, client)

    st.subheader("Auto-Generated Insights")
    st.text_area("Insight Summary", insights, height=300)

    st.subheader("Ask a Question")
    user_q = st.text_input("Your Question")
    if user_q:
        result = query_chatbot_full_context(user_q, docs, client)
        st.markdown("**Response:**")
        st.write(result)

else:
    st.warning("Please upload a CSV file and provide a valid OpenAI API key.")