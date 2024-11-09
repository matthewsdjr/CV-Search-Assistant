import streamlit as st
from typing import Dict, List, Any
import datetime
from pprint import pprint

# Importar todas las dependencias necesarias
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool, initialize_agent
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.chains import RetrievalQA

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Búsqueda de CVs",
    page_icon="🔍",
    layout="wide"
)

# Template personalizado para el QA de CVs
cv_qa_template = """
Eres un asistente especializado en búsqueda y análisis de perfiles profesionales para procesos de selección. 
Tu objetivo es ofrecer respuestas detalladas y precisas basadas en los criterios proporcionados.

Contexto del CV:
{context}

Pregunta: {question}

Instrucciones específicas:
1. Evalúa los requisitos y criterios mencionados en la pregunta
2. Calcula la experiencia total en meses y años cuando sea relevante
3. Clasifica al candidato como "cumple" o "se aproxima" según los criterios
4. Presenta la información en formato estructurado incluyendo:
   - Nombre
   - Contacto (email, teléfono)
   - Experiencia relevante
   - Score de similitud con los requisitos
5. Si la información es ilegible o incompleta, indícalo claramente

Respuesta:
"""

# Variables globales para almacenar componentes inicializados
qa_chain = None
docsearch = None

def process_cv_search(query: str, limit: int = 10) -> dict:
    """
    Función para procesar búsquedas de CVs
    """
    try:
        result = qa_chain.invoke({
            "query": query,
            "limit": limit
        })
        print(f'\n### resulta: {result}')
        # Procesar y estructurar la respuesta
        processed_result = {
            "matches": [],
            "metadata": {
                "total_results": len(result["source_documents"]),
                "query": query,
                "limit": limit
            }
        }
        
        # Procesar cada documento fuente
        for doc in result["source_documents"][:limit]:
            processed_result["matches"].append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": doc.metadata.get("score", 0)
            })
            
        return processed_result
    except Exception as e:
        return {"error": str(e)}

# Función para inicializar las variables de sesión
def init_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'agent' not in st.session_state:
        initialize_search_agent()

def initialize_search_agent():
    global qa_chain, docsearch
    
    # Cargar variables de entorno
    load_dotenv("C:/Users/matth/OneDrive/KAIZENLAB/ASSISTANT_SCRIPT_PYTHON/.env")
    
    api_key = os.environ['OPENAI_API_KEY']
    pinecone_key = os.environ['PINECONE_API_KEY']
    
    # Inicializar clientes
    pc = Pinecone(api_key=pinecone_key)
    index_name = 'text-embedding-ada-002'
    
    # Configurar embeddings y vectorstore
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Configurar LLM y memoria
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4",
        temperature=0.0
    )
    
    conversational_memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        k=10,
        return_messages=True,
        llm=llm,
        max_token_limit=2000
    )
    
    # Configurar el prompt y qa_chain
    prompt = PromptTemplate(
        template=cv_qa_template,
        input_variables=["context", "question"],
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    # Configurar herramientas y agente
    tools = [
        Tool(
            name='SearchCVs',
            func=process_cv_search,
            description=(
                'Utiliza esta herramienta para buscar y analizar CVs basado en criterios específicos. '
                'La herramienta devolverá perfiles relevantes con información detallada incluyendo '
                'experiencia, habilidades y datos de contacto(NOMBRE, EMAIL, TELEFONO). Necesito conocer el source de cada documento.'
                'En la respuesta final también incluirá la información de los posibles candidatos'
            )
        )
    ]
    
    st.session_state.agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

def search_candidates(query: str) -> Dict:
    """Función principal para buscar candidatos"""
    try:
        response = st.session_state.agent.invoke({
            "input": query
        })
        
        # Agregar la conversación al historial
        st.session_state.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.datetime.now()
        })
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response['output'],
            "timestamp": datetime.datetime.now()
        })
        
        return {
            "status": "success",
            "response": response,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    st.title("🔍 Búsqueda Inteligente de CVs")
    
    # Inicializar estado de la sesión
    init_session_state()
    
    # Crear el área de búsqueda
    with st.container():
        st.markdown("### 🔎 Nueva Búsqueda")
        query = st.text_area(
            "Ingrese los criterios de búsqueda:",
            placeholder="Ejemplo: Busco candidatos con experiencia en minería y conocimientos de seguridad industrial...",
            height=100
        )
        
        if st.button("Buscar", type="primary"):
            with st.spinner("Buscando candidatos..."):
                result = search_candidates(query)
                if result["status"] == "error":
                    st.error(f"Error en la búsqueda: {result['error']}")
    
    # Mostrar el historial de conversación
    st.markdown("### 💬 Historial de Conversación")
    
    for message in reversed(st.session_state.conversation_history):
        with st.container():
            col1, col2 = st.columns([1, 8])
            
            # Mostrar el ícono según el rol
            with col1:
                icon = "👤" if message["role"] == "user" else "🤖"
                st.markdown(f"### {icon}")
            
            # Mostrar el contenido del mensaje
            with col2:
                st.markdown(f"**{message['role'].title()}**")
                st.markdown(message["content"])
                st.markdown(f"*{message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
            
            st.divider()

if __name__ == "__main__":
    main()