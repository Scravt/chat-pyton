# Traer el archivo .env
import os
from dotenv import load_dotenv
# Cargar variables de entorno
load_dotenv()
# Acceder a la variable de entorno COHERE_API_KEY
cohere_api_key = os.getenv("COHERE_API_KEY")

# Importar la clase ChatCohere
from langchain_cohere import ChatCohere
llm = ChatCohere()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

# Crear el cargador de documentos PDF
loader = PyPDFLoader("./pdf/Dialnet-AplicacionDeLaTeoriaFundamentadaGroundedTheoryAlEs-2499458.pdf")
docs = loader.load()

# Proceso de embedding
from langchain_cohere.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings()

# Almacén de vectores local simple, FAISS
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Crear cadena de recuperación
prompt = ChatPromptTemplate.from_template("""responde desde el context y di elemental mi querido watson en cada respuesta, diq ue no puedes responder a nada que no este relacionado con el contex:
<context>
{context}
</context>
Question: {input}""")

# Crear cadena de documentos
document_chain = create_stuff_documents_chain(llm, prompt)

# Convertir la base de vectores a un recuperador (retriever)
retriever = vector.as_retriever()

# Crear la cadena de recuperación
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def call_api_IA(user_input):
    response = retrieval_chain.invoke({"input": user_input})
    return response
