from docx import Document
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import getSampleStyleSheet


import streamlit as st


from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.chains import ConversationChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

def get_custom_prompt_template(recommendation_engine):
    if recommendation_engine == "profilbasierte":        
        custom_prompt_template= """         
        Projektexposé muss im Bereich Informatik, maschinelles Lernen und künstliche Intelligenz im Allgemeinen sein. Die Studierenden geben möglicherweise keine präzise Beschreibung und Ideen an. Sie möchten einfach vorgeschlagene Themen mit Datensätzen haben.        
        Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mindestens 10 verschiedene Datensätze zu den Themen Computer Vision, Natural Language Processing oder Zeitreihen bereitstellen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten die Datensätze nach Thema gruppieren.         
        4. Die Antwort sollte den Kaggle-Datensatzlink enthalten.                  Dies ist der Ausschnitt des Textes, für den Sie Datensätze finden müssen:         
        {question}                  
        Nachfolgend ist der Kontext:         
        {context}                           
        """
    elif recommendation_engine == "expertenbasierte":
        custom_prompt_template= """         
        Basierend auf dem Projektexposé kombiniert KaggleGPT aktuelle Trends und Themen in den Bereichen und schlägt anspruchsvolle Ideen mit Datensätzen vor. Die Ausgabe hier ist für gute Studierende gedacht, die anspruchsvolle Ideen verfolgen möchten.        
        Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mehrere aktuelle interessante Trends zusammenfassen, um Studierende bei der Arbeit mit anspruchsvollen Datensätzen zu überzeugen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten mindestens 8 verschiedene Datensätze bereitstellen.         
        4. Sie sollten die Datensätze nach Größe und Benutzbarkeitsbewertung sortieren. Je größer die Größe und die Benutzbarkeitsbewertung, desto schwieriger ist es, mit diesen Datensätzen zu arbeiten.         
        5. Sie sollten zusätzliche Fortschritte wie die Anforderung an die Studierenden, die Verwendung leistungsstarker Rechensysteme oder Cloud-Plattformen zum Arbeiten mit großen Datensätzen in Betracht zu ziehen, die Entwicklung eines ausführbaren Prototyps oder das Bereitstellen einer Demo bereitstellen.         
        6. Die Antwort sollte den Kaggle-Datensatzlink enthalten. 
        
        Nachfolgend ist der Kontext:         
        {context}         
        Dies ist der Ausschnitt des Textes, für den Sie Datensätze finden müssen:         
        {question}                
        """
    elif recommendation_engine == "wissensbasierte":
        custom_prompt_template = """         
        Die Ausgaben basieren ausschließlich auf den Masterprogrammen und Lehrplänen mit festgelegten Lernergebnissen. Wie ein reguläres Projekt aussehen sollte. 
        Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mindestens 10 verschiedene Datensätze zu den Themen Computer Vision, Natural Language Processing oder Zeitreihen bereitstellen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten die Datensätze nach Thema gruppieren.         
        4. Sie sollten die Datensätze nach viewCount und voteCount sortieren. Je größer die viewCount und voteCount, desto beliebter sind diese Datensätze zum Arbeiten.         
        5. Die Antwort sollte den Kaggle-Datensatzlink enthalten.                  Nachfolgend ist der Kontext:         
        {context}         
        Dies ist der Ausschnitt des Textes, für den Sie Datensätze finden müssen:         
        {question}      
        """
    elif recommendation_engine == "multi-kriterienbasierte":
        custom_prompt_template = """         
        Die kombinierte Empfehlung berücksichtigt andere Metainformationen, wie lange die Dauer der Abschlussarbeit ist. Ist das Thema für den eingeschränkten Zeitrahmen geeignet? Investieren die Studierenden in GPU-Arbeitsstationen oder Cloud-Computing, um Experimente durchzuführen? Möchten die Studierenden aus den Ergebnissen eine Konferenz- und Zeitschriftenabgabe machen? KaggleGPT könnte die Studierenden fragen, ob sie die erforderlichen Kriterien haben. Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mehrere aktuelle interessante Trends zusammenfassen, um Studierende bei der Arbeit mit anspruchsvollen Datensätzen zu überzeugen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten mindestens 8 verschiedene Datensätze bereitstellen.        
        4. Sie sollten die Datensätze nach Größe, Benutzbarkeitsbewertung, viewCount und voteCount sortieren. Je größer die Zahlen, desto herausfordernder ist es, mit diesen Datensätzen zu arbeiten.         
        5. Die Antwort sollte den Kaggle-Datensatzlink enthalten.         
        6. Sie sollten erwähnen, dass das Einreichen eines Forschungspapiers sehr empfohlen wird.         
        7. Sie sollten zusätzliche Fortschritte bereitstellen, z. B. die Anforderung an die Studierenden, leistungsstarke Rechensysteme oder Cloud-Plattformen zum Arbeiten mit großen Datensätzen zu verwenden. Die Studierenden müssen auch einen ausführbaren Prototyp entwickeln oder eine Demo bereitstellen.                  
        Nachfolgend ist der Kontext:         
        {context}         
        Dies ist der Ausschnitt des Textes, für den Sie Datensätze finden müssen:         
        {question}
        """  
    return custom_prompt_template 
       

def get_llm_response(user_query,open_ai_llm,retriever,memory,recommendation_engine): 
    template = get_custom_prompt_template(recommendation_engine)
    # Create a PromptTemplate instance with your custom template
    custom_prompt = PromptTemplate(
        template= template,
        input_variables=["context", "question"],
    )
    # Use your custom prompt when creating the ConversationalRetrievalChain

    qa = ConversationalRetrievalChain.from_llm(
        open_ai_llm,
        verbose=True,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )        
    response = qa({"question": user_query})['answer']
    return response

def download_the_conversation(messages, filename='gesprach.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []

    title_style = getSampleStyleSheet()["Title"]
    title_data = [
        Paragraph('<b><u>KaggleGPT German : Ein multi-kriterien LLM-basiertes Empfehlungssystem zur effizienten Untersuchung von Datensätzen in Projekten zum maschinellen Lernen</u></b>', title_style),
        Paragraph('<br/>', title_style)  # Add two break lines before user response
    ]
    story.extend(title_data)
    
    # Add dialogues to the PDF
    dialogue_style_user = getSampleStyleSheet()["BodyText"]
    dialogue_style_user.fontName = 'Helvetica'  # Change font to Helvetica for better readability
    dialogue_style_assistant = getSampleStyleSheet()["BodyText"]
    dialogue_style_assistant.fontName = 'Helvetica'
    dialogue_style_assistant.alignment = 0  # Align left for assistant's messages

    for message in messages:
        role = message['role']
        content = message['content']

        if role == 'user':
            dialogue_data = [
                Paragraph(f'<b>User:</b> {content}', dialogue_style_user),
                Paragraph('<br/>', dialogue_style_user)  # Add a break line between user and assistant messages
            ]
        else:
            dialogue_data = [
                Paragraph(f'<b>Kaggle Recommender Engine:</b> {content}', dialogue_style_assistant),
                Paragraph('<br/>', dialogue_style_assistant)
            ]
        story.extend(dialogue_data)
    doc.build(story)
    return filename

def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_pdf(file):
    doc = PdfReader(file)
    text = ""
    for page in doc.pages:
        text += page.extract_text()
    return text


def initial_response_query_and_answer(query, bots_response,recommendation_engine):
    response_text = f"Sie sind eine {recommendation_engine} Kaggle-Datensatzempfehlungs-Engine \n Benutzeranfrage: {query}\n\nEmpfohlene Datensatzinformationen: {bots_response}"
    return response_text

def conversation_object(recommendation_engine):  
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,return_messages=True)
        
    if recommendation_engine == "profilbasierte":
        template = """
        Sie sind ein Datensatz-Empfehlungssystem, das den Studierenden die erforderlichen Datensätze zur Verfügung stellt und alle möglichen Fragen basierend auf dem Kontext und der Historie des Chats beantwortet. Die Studierenden sind Masterstudierende in Maschinellem Lernen, Data Science und Künstlicher Intelligenz.
        
        Sie erkennen die Eingangssprache des Benutzers aus dem Kontext und geben die Antwort in derselben Sprache aus.
        
        Hier ist der Kontext:
        """
        
    elif recommendation_engine == "expertenbasierte":
        template = """
        Sie sind ein Datensatz-Empfehlungssystem, das den Studierenden die erforderlichen Datensätze zur Verfügung stellt und basierend auf dem Kontext und der Chat-Historie alle möglichen Fragen beantwortet. Sie kombinieren Ihr aktuelles Wissen mit dem Kontext und stellen herausfordernde Datensätze bereit. Die herausfordernden Datensätze werden durch ihre Größe und Benutzerfreundlichkeit definiert. Je größer die Größe und die Benutzerfreundlichkeit, desto anspruchsvoller sind die Datensätze. Bitte stellen Sie externe Datensätze bereit, falls erforderlich.

        Sie erkennen die Eingangssprache des Benutzers aus dem Kontext und geben die Antwort in derselben Sprache aus.

        Hier ist der Kontext:
        """
    
    elif recommendation_engine == "wissensbasierte":
        template = """
        Sie sind ein Datensatz-Empfehlungssystem, das den Studierenden die erforderlichen Datensätze zur Verfügung stellt und basierend auf dem Kontext und der Historie des Chats alle möglichen Fragen beantwortet. Die Studierenden sind Masterstudenten im Bereich Maschinelles Lernen, Data Science und Künstliche Intelligenz. Sie befinden sich in ihrem letzten Studienjahr und absolvieren die erforderlichen Vorkurse. Sie stellen Datensätze basierend auf den Informationen zu viewCount und voteCount bereit. Je höher der viewCount und voteCount, desto beliebter ist es, mit diesen Datensätzen zu arbeiten.

        Sie erkennen die Eingangssprache des Benutzers aus dem Kontext und geben die Antwort in derselben Sprache aus.

        Hier ist der Kontext:
        """
    
    elif recommendation_engine == "multi-kriterienbasierte":
        template = """
        Sie sind ein Datensatz-Empfehlungssystem, das den Studierenden die erforderlichen Datensätze zur Verfügung stellt und basierend auf dem Kontext und der Historie des Chats alle möglichen Fragen beantwortet. Sie kombinieren Ihr aktuelles Wissen mit dem Kontext, bieten herausfordernde Datensätze an, die auf Größe, Benutzerfreundlichkeit, viewCount und voteCount fokussiert sind. Je größer die Zahlen, desto schwieriger ist es, mit diesen Datensätzen zu arbeiten. Bitte stellen Sie externe Datensätze bereit, falls erforderlich. Es wäre hilfreich zu erwähnen, dass die Veröffentlichung nach den Experimenten ein Muss ist. Außerdem wäre es hilfreich, externe Basislinien auf Basis Ihres aktuellen Wissens bereitzustellen.
        
        Sie erkennen die Eingangssprache des Benutzers aus dem Kontext und geben die Antwort in derselben Sprache aus.

        Hier ist der Kontext:
        """
          
    system_msg_template = SystemMessagePromptTemplate.from_template(template=template)
    
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    conversation = ConversationChain(memory=st.session_state.buffer_memory,   prompt=prompt_template, llm=st.session_state.open_ai_llm, verbose=True)
    
    return conversation

