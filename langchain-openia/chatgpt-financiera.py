import os
import sys

import json 
# import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)

# if os.path.exists("../data"):
#     loader = PyPDFDirectoryLoader("./data/")
#     documents = loader.load()
#     print(len(documents))

deembeddings = OpenAIEmbeddings()
# Vectoriza los documentos
# vectordb = Chroma.from_documents(
#     documents, embeddings)

# Aplica el modelo de ChatGPT para conversacion
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)


# Construye un templeta de prompt
template = """Utiliza todas las piezas del Texto y de la Historia para responder.
 Debes realizar las siguiente acciones:
 1. Resumir el texto en maximo 1 oracion delimitado entre <>.
 2. Si existen nombres de personas separa el nombre de pila y etiquetarlo como <nombre> y el o los apellidos como <apellido>.
 3. Si existe parte del texto numeros de 7 u 8 caracteres etiquetarlo como <dni>.
 4. Extraer la intencion en un maximo de 2 palabras y etiquetalo como <intencion>.
 5. Generar una respuesta con una frase amigable en una etiqueta <frase>.
 6. Si cononces el nombre de la persona incluye su nombre en la respuesta.
 7. La resuesta no puede ser mas de 2 oraciones.
 8. Devoler la respuesta en un json con el siguiente formato:

 Texto: {query}
 Resumen: <resumen>
 Intencion: <intencion>
 Nombre: <nombre>
 Apellido: <apellido>
 Documento: <dni>
 Respuesta: <frase>

Texto: {query}
Historia: {chat_history}"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    # result = chain({"query": query})
    result = chain({"query": query, "chat_history": chat_history})
    print(result['text'])
    dato= json.loads(result['text'])
    print(dato["Respuesta"])

    chat_history.append((query, dato['Respuesta']))
    query = None
