import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader

from flask import Flask, request, jsonify
from langchain_groq.chat_models import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import (create_retrieval_chain,
                              create_history_aware_retriever, create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Retrieve API keys
os.getenv("GROQ_API_KEY")
os.getenv("HUGGINGFACE_API_KEY")

# Function to extract text from PDF files
def get_pdf_text(pdf_dir):
    text = ""

    for docs in os.listdir(pdf_dir):
        docs_path = os.path.join(pdf_dir, docs)

        with open(docs_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vectorstore(text_chunks, dbname="faq_index"):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local(dbname)

    return vectorstore

# Prompts for contextualization and QA
contextualization_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

# Create prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualization_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Function to generate response using RAG chain
def generate_response(llm, retriever, chat_history, input):    
    history_aware_retriever = create_history_aware_retriever(
                                        llm, retriever, contextualize_q_prompt
                                    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    response = rag_chain.invoke({"input": input, "chat_history":chat_history})
    chat_history.extend([input, response["answer"]])
    return (chat_history, response['answer'])

# Initialize LLM and retrieve documents
llm = ChatGroq()
pdf_dir = './Faq docs'
text = get_pdf_text(pdf_dir)
chunks = get_text_chunks(text)
vectorstore = get_vectorstore(chunks)
retriever = vectorstore.as_retriever()

# Create Flask app
app = Flask("Generate")

# Define the route for generating responses
@app.route("/promptresp", methods=["POST"])
def predict():
    modelinput = request.get_json()
    prompt, chat_history = modelinput["prompt"], modelinput["chat_history"]
    chat_history, response = generate_response(llm, retriever, chat_history, prompt)
    output = jsonify({"chat_history": chat_history, "response": response})
    
    return output

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=9696)
