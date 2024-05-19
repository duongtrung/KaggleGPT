import time
from dotenv import load_dotenv
import streamlit as st
from streamlit import session_state as ss
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval
from utils import extract_text_from_pdf, prompt_template, download_conversation

# Load Environment Variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI()

# Initialsing Session Cariables
if 'stream' not in ss:
    ss.stream = None

if "messages" not in ss:
    ss.messages = []

if "is_pdf_file_uploaded" not in ss:
    ss.is_pdf_file_uploaded = False 
            
if "uploaded_pdf_file" not in ss:
    ss.uploaded_pdf_file = None
 
if "is_initial_response_generated" not in ss:
    ss.is_initial_response_generated = False
           
# Supporting Functions
def data_streamer():
    """
    That stream object in ss.stream needs to be examined in detail to come
    up with this solution. It is still in beta stage and may change in future releases.
    """
    for response in ss.stream:
        if response.event == 'thread.message.delta':
            value = response.data.delta.content[0].text.value
            yield value
            time.sleep(0.001)

@st.cache_resource
def vector_store_creation():
    vector_store = client.beta.vector_stores.create(name="Kaggle Recommendation System")
    file_paths = ["knowledge_base/kaggle_datasets.pdf"]
    file_streams = [open(path, "rb") for path in file_paths]
    client.beta.vector_stores.file_batches.upload_and_poll(vector_store_id=vector_store.id, files=file_streams)
    return vector_store
    
@st.cache_resource
def init_assistant(_vector_store,prompt_instructions):
    """Define client and assistant"""
      
    assistant = client.beta.assistants.create(
        name="Kaggle Dataset Recommedion System",
        instructions=prompt_instructions,
        tools=[{"type": "file_search"}],
        model="gpt-4o",
        tool_resources={"file_search": {"vector_store_ids": [_vector_store.id]}}
    )
    return client, assistant

def upload_pdf_file():
    with st.sidebar:    
        with st.form("my-form"):
            ss.uploaded_pdf_file = st.file_uploader("Laden Sie Ihr Projekt-Exposé hoch: ", type=["pdf"]) 
            
            submitted = st.form_submit_button("Holen Sie sich die von Kaggle empfohlenen Datensätze")

        if submitted and ss.uploaded_pdf_file is not None:
            st.success("Datei erfolgreich hochgeladen!")
            ss.is_pdf_file_uploaded = True

def create_new_session():
    with st.sidebar:   
        with st.form("new_session_form"):
            st.success(f"Datei bereits hochgeladen : {ss.uploaded_pdf_file.name}")
            submitted = st.form_submit_button("Eine neue Sitzung erstellen.")
        if submitted:
            with st.spinner("Neue Sitzung wird erstellt...."):
                ss.uploaded_pdf_file = None
                ss.is_pdf_file_uploaded = False
                ss.messages = []
                ss.is_initial_response_generated = False
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                
def display_app_instructions():
    with st.sidebar:
        st.write("""
                 Hilfe zum System:
                 
                 Die Studierenden schreiben einen Entwurf für das Projektexposé.
                 - Wählen Sie Themen und Interessen, die mit den Inhalten des Masterstudiengangs zusammenhängen.
                 - Reflektieren Sie Ihr Wissen auf 1-2 Seiten.
                 - Entwickeln Sie eine erste Idee für ein mögliches Thema und Forschungsfragen.
                 - Finden Sie geeignete Datensätze.
                 """) 

# Main function
def main():
    
    st.set_page_config(page_title="KaggleGPT",page_icon='🤖',layout='centered',initial_sidebar_state='expanded')
    st.header("KaggleGPT German")
    st.subheader("Ein multi-kriterien LLM-basiertes Empfehlungssystem zur effizienten Untersuchung von Datensätzen in Projekten zum maschinellen Lernen")
    
    vector_store = vector_store_creation()
    
    with st.sidebar:
        st.sidebar.title("Benutzereinstellungen") 
        
        recommendation_engine = st.radio(
            "Empfehlungsgebung:",
            ('profilbasierte', 'expertenbasierte', 'wissensbasierte','multi-kriterienbasierte'))
            
        if recommendation_engine == 'profilbasierte':
            st.write('Sie haben die profilbasierte Empfehlungsgebung gewählt.')
        
        elif recommendation_engine == 'expertenbasierte':
            st.write('Sie haben die expertenbasierte Empfehlungsgebung gewählt.')
        
        elif recommendation_engine == 'wissensbasierte':
            st.write('Sie haben die wissensbasierte Empfehlungsgebung gewählt.') 
            
        elif recommendation_engine == 'multi-kriterienbasierte':
            st.write('Sie haben die multi-kriterienbasierte Empfehlungsgebung gewählt.')
    
    # Fetching the prompt template
    prompt_instructions = prompt_template(recommendation_engine)
    
    if not ss.is_pdf_file_uploaded:  
        upload_pdf_file()
    else:
        create_new_session()
    
    # Initialize openai assistant
    client, assistant = init_assistant(vector_store,prompt_instructions)
    
    # Display Messages
    for message in ss.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if len(ss.messages)  > 0:        
        with st.sidebar:         
            if st.button("Konversation herunterladen"):
                st.success("PDF erfolgreich erstellt! Klicken Sie auf die Schaltfläche unten, um es herunterzuladen.")
                # Create PDF and get its filename
                pdf_filename = download_conversation(ss.messages)
                # Provide the generated PDF for download
                st.download_button(
                    label="PDF herunterladen",
                    data=open(pdf_filename, 'rb').read(),
                    file_name=pdf_filename,
                    mime='application/pdf') 
    
    # Display App Instructions
    display_app_instructions()
                   
    # Reading PDF text and generate recommended Kaggle Datasets             
    if not ss.is_initial_response_generated: 
        if ss.is_pdf_file_uploaded:
            # INITIAL QUERY  
            with st.spinner("Datei wird gelesen...."):
                user_input_pdf_text = extract_text_from_pdf(ss.uploaded_pdf_file)
                ss.messages.append({"role": "user", "content": user_input_pdf_text})
                with st.chat_message("user"):
                    st.markdown(user_input_pdf_text)
        
            msg_history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
            with st.spinner("Bitte warten....Empfehlungen für Kaggle-Datensätze abrufen......"):
                try:
                    ss.stream = client.beta.threads.create_and_run(
                        assistant_id=assistant.id,        
                        thread={
                            "messages": msg_history
                        },
                        stream=True
                    )
                    with st.chat_message("assistant"):
                        response = st.write_stream(data_streamer)
                        ss.messages.append({"role": "assistant", "content": response}) 
                        st.session_state.is_initial_response_generated = True        
                except Exception as e:
                    st.error("Es scheint ein Fehler mit Ihrem OpenAI API-Zugriffsschlüssel aufgetreten zu sein. Bitte geben Sie den korrekten OpenAI-Zugriffsschlüssel ein.")   
        else:
            st.info("Bitte laden Sie Ihr Thesendokument in die linke Seitenleiste hoch und klicken Sie auf die Schaltfläche 'Absenden'.")
    
    # Chat Section
    if ss.is_initial_response_generated:
        if prompt := st.chat_input("Stellen Sie mir eine Frage..."):
            ss.messages.append({"role": "user", "content": prompt})
            
            # Prompt from User
            with st.chat_message("user"):
                st.markdown(prompt)
            
            msg_history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
            
            # Response from LLM
            try:
                with st.spinner("Antwort abrufen...."):
                    ss.stream = client.beta.threads.create_and_run(
                        assistant_id=assistant.id,        
                        thread={
                            "messages": msg_history
                        },
                        stream=True
                    )
                    
                    with st.chat_message("assistant"):
                        response = st.write_stream(data_streamer)
                        ss.messages.append({"role": "assistant", "content": response})
                        
            except Exception as e:
                st.error("Es scheint ein Problem beim Abrufen der Antwort aufgetreten zu sein. Bitte starten Sie die Sitzung neu.")

            
if __name__ == '__main__':
    main()