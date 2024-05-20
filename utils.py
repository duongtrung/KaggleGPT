from docx import Document
from markdown2 import markdown
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def prompt_template(recommendation_engine):
    if recommendation_engine == "profilbasierte":
        custom_prompt_template = """
        Projektexposé muss im Bereich Informatik, maschinelles Lernen und künstliche Intelligenz im Allgemeinen sein. Die Studierenden geben möglicherweise keine präzise Beschreibung und Ideen an. Sie möchten einfach vorgeschlagene Themen mit Datensätzen haben.        
        Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mindestens 10 verschiedene Datensätze zu den Themen Computer Vision, Natural Language Processing oder Zeitreihen bereitstellen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten die Datensätze nach Thema gruppieren.         
        4. Die Antwort sollte den Kaggle-Datensatzlink enthalten.                                
        """
    
    elif recommendation_engine == "expertenbasierte": 
        custom_prompt_template = """
        Basierend auf dem Projektexposé kombiniert KaggleGPT aktuelle Trends und Themen in den Bereichen und schlägt anspruchsvolle Ideen mit Datensätzen vor. Die Ausgabe hier ist für gute Studierende gedacht, die anspruchsvolle Ideen verfolgen möchten.        
        Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mehrere aktuelle interessante Trends zusammenfassen, um Studierende bei der Arbeit mit anspruchsvollen Datensätzen zu überzeugen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten mindestens 8 verschiedene Datensätze bereitstellen.         
        4. Sie sollten die Datensätze nach Größe und Benutzbarkeitsbewertung sortieren. Je größer die Größe und die Benutzbarkeitsbewertung, desto schwieriger ist es, mit diesen Datensätzen zu arbeiten.         
        5. Sie sollten zusätzliche Fortschritte wie die Anforderung an die Studierenden, die Verwendung leistungsstarker Rechensysteme oder Cloud-Plattformen zum Arbeiten mit großen Datensätzen in Betracht zu ziehen, die Entwicklung eines ausführbaren Prototyps oder das Bereitstellen einer Demo bereitstellen.         
        6. Die Antwort sollte den Kaggle-Datensatzlink enthalten. 
        """
        
    elif recommendation_engine == "wissensbasierte":
        custom_prompt_template = """
        Die Ausgaben basieren ausschließlich auf den Masterprogrammen und Lehrplänen mit festgelegten Lernergebnissen. Wie ein reguläres Projekt aussehen sollte. 
        Ihre Aufgaben sind wie folgt:         
        1. Sie sollten mindestens 10 verschiedene Datensätze zu den Themen Computer Vision, Natural Language Processing oder Zeitreihen bereitstellen.         
        2. Sie sollten die Ergebnisse in einer Tabelle zur einfachen Ansicht anzeigen.         
        3. Sie sollten die Datensätze nach Thema gruppieren.         
        4. Sie sollten die Datensätze nach viewCount und voteCount sortieren. Je größer die viewCount und voteCount, desto beliebter sind diese Datensätze zum Arbeiten.         
        5. Die Antwort sollte den Kaggle-Datensatzlink enthalten.
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
        """
    return custom_prompt_template
             
             
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_pdf(file):
    doc = PdfReader(file)
    text = ""
    for page in doc.pages:
        text += page.extract_text()
    return text


def download_conversation(messages, filename='kaggle_dataset_recommendation_system.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []

    # Use Times-Bold which is available by default in ReportLab
    title_style = getSampleStyleSheet()["Title"]
    title_style.fontName = 'Times-Bold'
    title_data = [
        Paragraph('<b><u>KaggleGPT German : Ein multi-kriterien LLM-basiertes Empfehlungssystem zur effizienten Untersuchung von Datensätzen in Projekten zum maschinellen Lernen</u></b>', title_style),
        Spacer(1, 12)  # Add space instead of break line for better control
    ]
    story.extend(title_data)

    # Define custom styles
    user_style = ParagraphStyle(
        name='UserStyle',
        parent=getSampleStyleSheet()["BodyText"],
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.black,
        backColor=colors.white
    )
    assistant_style = ParagraphStyle(
        name='AssistantStyle',
        parent=getSampleStyleSheet()["BodyText"],
        fontName='Times-Roman', 
        fontSize=12,
        textColor=colors.black,
        backColor=colors.lightgrey,
        borderPadding=(6, 6, 6, 6),
        borderRadius=8
    )

    for message in messages:
        role = message['role']
        content = markdown(message['content'])  # Convert markdown to HTML

        if role == 'user':
            dialogue_data = [
                Paragraph(f'<b>User:</b> {content}', user_style),
                Spacer(1, 6)  # Add a small space between messages
            ]
        else:
            dialogue_data = [
                Paragraph(f'<b>Kaggle Recommender Engine:</b> {content}', assistant_style),
                Spacer(1, 6)
            ]
        story.extend(dialogue_data)
        story.append(Spacer(1, 12))  # Add space between interactions

    doc.build(story)
    return filename