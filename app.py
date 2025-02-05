
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-openAI key" 
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import threading

def create_knowledge_base():
   
    loader = WebBaseLoader(["https://brainlox.com/courses/category/technical"])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
  
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

app = Flask(__name__)
api = Api(app)

vectorstore = create_knowledge_base()


qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.7),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        try:
            response = qa.run(user_message)
            return jsonify({"response": response, "status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


api.add_resource(Chatbot, "/chat")


def run_app():
    app.run(debug=True, use_reloader=False)

threading.Thread(target=run_app).start()
import requests

url = "http://127.0.0.1:5000/chat"  

data = {"message": "What courses are available?"}
response = requests.post(url, json=data)

print(response.json())  
