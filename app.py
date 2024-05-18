import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from utils import get_llm_response, download_the_conversation,read_pdf,initial_response_query_and_answer,conversation_object


def vector_db_retriever():
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local("vectorstore/db_faiss", embeddings,allow_dangerous_deserialization = True)
    st.session_state.loaded_db = vectordb
    retriever = vectordb.as_retriever(search_kwargs={"k": 12})
    return retriever

def sidebar_settings():
    with st.sidebar:   
        if len(st.session_state.messages) >= 1:
            if st.button("Konversation herunterladen"):
                st.success("PDF erfolgreich erstellt! Klicken Sie auf die Schaltfläche unten, um es herunterzuladen.")
                # Create PDF and get its filename
                pdf_filename = download_the_conversation(st.session_state.messages)
                # Provide the generated PDF for download
                st.download_button(
                label="Download PDF",
                data=open(pdf_filename, 'rb').read(),
                file_name="gesprach.pdf",
                mime='application/pdf')   
                
            if st.button("Neues Gespräch"):
                st.session_state.messages = []   
                st.session_state.uploaded_file = st.empty()
                
        st.write("""
                 Hilfe zum System:
                 
                 Die Studierenden schreiben einen Entwurf für das Projektexposé.
                 - Wählen Sie Themen und Interessen, die mit den Inhalten des Masterstudiengangs zusammenhängen.
                 - Reflektieren Sie Ihr Wissen auf 1-2 Seiten.
                 - Entwickeln Sie eine erste Idee für ein mögliches Thema und Forschungsfragen.
                 - Finden Sie geeignete Datensätze.
                 """)       

def init_page():
    st.header("KaggleGPT German")
    st.subheader("Ein multi-kriterien LLM-basiertes Empfehlungssystem zur effizienten Untersuchung von Datensätzen in Projekten zum maschinellen Lernen")
    st.sidebar.title("Benutzereinstellungen")
        
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []   
        

def upload_pdf_file():
    with st.sidebar: 
        if "is_pdf_file_uploaded" not in st.session_state:
                st.session_state.is_pdf_file_uploaded = False
                
        with st.form("my-form"):
            st.session_state.uploaded_file = st.file_uploader("Laden Sie Ihr Projekt-Exposé hoch: ", type=["pdf"]) 
            
            submitted = st.form_submit_button("Holen Sie sich die von Kaggle empfohlenen Datensätze")

        if submitted and st.session_state.uploaded_file is not None:
            st.success("Datei erfolgreich hochgeladen!")
            st.session_state.is_pdf_file_uploaded = True
    
            # do stuff with your uploaded file
        return st.session_state.uploaded_file 
    
def main():
    st.set_page_config(
                page_title="KaggleGPT",
                page_icon='🤖',
                layout='centered',
                initial_sidebar_state='expanded'
            )
    #language = select_language()
    init_page()
    # with st.sidebar: 
        # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        # os.environ["OPENAI_API_KEY"] = openai_api_key
        
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()
    # else: 
    with st.sidebar: 
        recommendation_engine = st.radio(
            "Empfehlungsgebung: ",
            ('profilbasierte', 'expertenbasierte', 'wissensbasierte','multi-kriterienbasierte'))
        st.session_state.recommendation_engine = recommendation_engine
        
        
        
    if recommendation_engine == 'profilbasierte':
        st.write('Sie haben die profilbasierte Empfehlungsgebung gewählt.')
              
    elif recommendation_engine == 'expertenbasierte':
        st.write('Sie haben die expertenbasierte Empfehlungsgebung gewählt.')
    elif recommendation_engine == 'wissensbasierte':
        st.write('Sie haben die wissensbasierte Empfehlungsgebung gewählt.')
    else:
        st.write('Sie haben die multi-kriterienbasierte Empfehlungsgebung gewählt.')
        
    upload_pdf_file()
    sidebar_settings() 
    
    if "retriever" not in st.session_state:
        retriever = vector_db_retriever()
        st.session_state.retriever = retriever
        
    if "memory" not in st.session_state:    
        memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True)
        st.session_state.memory = memory 
        
    if "open_ai_llm" not in st.session_state:
        llm = ChatOpenAI(model_name = "gpt-4", streaming=True)
        st.session_state.open_ai_llm = llm   
    
    if "is_initial_response_generated" not in st.session_state:
        st.session_state.is_initial_response_generated = False
    
    
    if not st.session_state.is_initial_response_generated: 
        
        if st.session_state.is_pdf_file_uploaded:
            # INITIAL QUERY  
            user_input_pdf_text = read_pdf(st.session_state.uploaded_file)
            st.session_state.messages.append({"role": "user", "content": user_input_pdf_text})
            
            with st.spinner("Bitte warten....Empfehlungen für Kaggle-Datensätze abrufen..."):
                try:
                    # INITIAL RESPONSE
                    response_from_llm = get_llm_response(user_input_pdf_text,st.session_state.open_ai_llm,st.session_state.retriever,st.session_state.memory,st.session_state.recommendation_engine)
            
                    st.session_state.response_from_llm = response_from_llm
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.response_from_llm})  
                    st.session_state.is_initial_response_generated = True        
                    intial_response_query_response_text = initial_response_query_and_answer(user_input_pdf_text,response_from_llm,st.session_state.recommendation_engine)
                    st.session_state.intial_response_query_response_text =  intial_response_query_response_text
                except Exception as e:
                    st.error("Seems there was an error with your OpenAI API Acess token key. Please enter the correct OpenAI Access key.")
                    
    if "conversation_object" not in st.session_state:
        st.session_state.conversation_object = conversation_object(st.session_state.recommendation_engine)
                    
    if st.session_state.is_initial_response_generated:   
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])                 
                
        user_query = st.chat_input("Stellen Sie mir eine Frage...")     
        if user_query:
            
            # QUERY           
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            #RESPONSE
            with st.chat_message("assistant"):
                with st.spinner("Antwort abrufen"):
                    context = st.session_state.intial_response_query_response_text
                    response_from_llm = st.session_state.conversation_object.predict(input=f"Context:\n {context} \n\n Query:\n{user_query}")
                    st.markdown(response_from_llm)
                    st.session_state.messages.append({"role": "assistant", "content": response_from_llm})          
        
if __name__ == "__main__":
    main()
