import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate

import constants
# REF  https://blog.devgenius.io/chat-with-document-s-using-openai-chatgpt-api-and-text-embedding-6a0ce3dc8bc8
# ref https://levelup.gitconnected.com/langchain-for-multiple-pdf-files-87c966e0c032

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)

loader = UnstructuredPDFLoader("data/doc1332618715.pdf")
#documents = loader.load_and_split(text_splitter)
documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# qindex = VectorstoreIndexCreator().from_loaders([loader])

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents, embeddings, persist_directory='persist')
vectordb.persist()
#vectordb = Chroma(texts, embeddings)
# vectordb = VectorstoreIndexCreator(
#     embedding=embeddings).from_documents(texts)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Build prompt
template = """Utiliza todas las piezas del contexto para responder a al pregunta. Si no contestas la pregutna, simplemente di que no lo sabes,no trataes de crear una respuesta. Utiliza tres oraciones como maximo. Manten las respuestas concisas. Siempre di "Gracias por preguntar" al final de cada respuesta. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
# chain=ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
# chain = RetrievalQA.from_chain_type(
#     llm,
#     return_source_documents=True,
#     retriever=vectordb.as_retriever(search_kwargs={"k": 1})
# )

# chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectordb.as_retriever(search_kwargs={"k": 1})
# )

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"query": query, "chat_history": chat_history})
    print(result['result'])

    chat_history.append((query, result['result']))
    query = None
