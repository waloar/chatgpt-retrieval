import os
import sys

import json 
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
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
template = """ Debes actuar como un agente de atencion a clientes. Debes dirigirte al interlocutor de una manera cordial y atenta, en lo posible por su nombre.
Utiliza todas las piezas del Texto y de la Historia para responder las preguntas. Si no contestas la pregutna, simplemente di que no lo sabes,no trates de crear una respuesta. Utiliza tres oraciones como maximo.
 Manten las respuestas concisas.
 Debes tener en consideracion:
 1. Si el interlocutor quiere sacar un prestamo, o desea pagar una deuda, o saber el estado de su cuenta, debes pedirle su nombre y su DNI.
 2. Si el interlocutor desea que le transfieras con un agente de ventas, o un supervisor, asegurate de tener el <dni> <nombre> y <apellido>, debes pedirle amablemente su nombre y su DNI y luego finalizar la conversacion.
 Ademas debes:
 1. Resumir el texto en maximo 1 oracion y etiquetalo como <resumen>.
 2. Si existen nombres de personas separa el nombre de pila y etiquetarlo como <nombre> y el o los apellidos como <apellido>.
 3. Si existe parte del texto numeros de 7 u 8 caracteres etiquetarlo como <dni> y formatearlo como xx.xxx.xxx .
 4. Extraer la intencion en un maximo de 2 palabras y etiquetalo como <intencion>.
 5. En todos los casos, genera una respuesta amigable con una frase amigable en una etiqueta <frase>.
 6. Si cononces el nombre de la persona incluye su nombre en la respuesta <frase>.
 7. La <frase> no puede ser mas de 2 oraciones.
 9. Devoler la respuesta en un json con el siguiente formato, donde <resumen>, <intencion>, <nombre>, <apellido>, <dni>, <frase> son los valores extraidos de los pasos anteriores. En el caso que no se pueda extraer algun valor, se debe devolver un string vacio.
 Resumen: <resumen>
 Intencion: <intencion>
 Nombre: <nombre>
 Apellido: <apellido>
 Documento: <dni>
 Respuesta: <frase>


 query: {query}
 chat_history : {chat_history}
 """
QA_CHAIN_PROMPT = PromptTemplate.from_template(role="Agente de Ventas", template=template)

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
    print(dato["frase"])

    chat_history.append((query, dato['frase']))
    query = None
