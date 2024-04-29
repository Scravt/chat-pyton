#traer el archivo .env
import os
from dotenv import load_dotenv
# Cargar variables de entorno
load_dotenv()
# Acceder a la variable de entorno COHERE_API_KEY
cohere_api_key = os.getenv("COHERE_API_KEY")


# Importar la clase ChatCohere
from langchain_cohere import ChatCohere
llm = ChatCohere()


from langchain_core.prompts import ChatPromptTemplate     # modelo para agregar pronto estructura llamada chain = prompt | llm 
prompt = ChatPromptTemplate.from_messages([
    ("system", "eres un profesor, que responde muy breve en 20 o 30 palabras y comienzas todas tu respuestas diciendo 'Elemental, querido Watson'"),
    ("user", "{input}")
])
 
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

##                     TRAER DOCUMENTOS DE INTERNET 
#aqui empezamos con la recuperacion de archivos y vectores apra contexto
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://es.wikipedia.org/wiki/Cactaceae")
docs = loader.load()



#proceso de ebmedding
from langchain_cohere.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings()


#almacén de vectores local simple, FAISS
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

print(vector)
#Ahora que tenemos estos datos indexados en un vectorstore, crearemos una cadena de recuperación.
#Esta cadena tomará una pregunta entrante, buscará documentos relevantes,
#luego pasará esos documentos junto con la pregunta original a un LLM y le pedirá que responda la pregunta original.

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Crear cadena de documentos
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Convertir tu base de vectores a un recuperador (retriever)
retriever = vector.as_retriever()

# Crear la cadena de recuperación
retrieval_chain = create_retrieval_chain(retriever, document_chain)

while True:
    user_input = input("You: ")
    
    # Si el usuario ingresa "exit", salir del bucle
    if user_input.lower() == "exit":
        break
    
    # Invocar la cadena de recuperación con la pregunta del usuario
    response = retrieval_chain.invoke({
        "input": user_input
    })
    
    # Imprimir la respuesta del modelo
    print("Model:", response["answer"])

