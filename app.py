"""
Pocket Lawyer - Indian Legal Assistant POC
RAG-based chatbot for legal questions with citations
"""

import os
import re
import time
import gradio as gr
from anthropic import Anthropic, RateLimitError, APIError
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="legal_docs",
    embedding_function=sentence_transformer_ef
)

SYSTEM_PROMPT = """You are a legal information assistant for Indian citizens. 

RULES:
1. Give SHORT, DIRECT answers (2-3 sentences maximum unless absolutely necessary)
2. Answer using only the provided context
3. Cite sources: [Source: Act Name, Section X]
4. Use simple language
5. If info not in context: "I don't have this information in my knowledge base."
6. End with: "Disclaimer: This is general information, not legal advice. Consult a lawyer."

Be concise and direct."""


def extract_section_info(text):
    """Extract section and chapter information from legal text"""
    sections = re.findall(r'(?:Section|SECTION)\s+(\d+[A-Z]?(?:\(\d+\))?)', text, re.IGNORECASE)
    chapters = re.findall(r'(?:Chapter|CHAPTER)\s+([IVX]+|\d+)', text, re.IGNORECASE)
    return {
        'sections': list(set(sections))[:5],
        'chapters': list(set(chapters))[:3]
    }


def clean_text(text):
    """Clean extracted PDF text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'[-_]{3,}', '', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.strip()


def smart_chunk_text(text, source_name, chunk_size=800, overlap=200):
    """Improved chunking strategy for legal documents with better section detection"""
    chunks = []
    text = clean_text(text)
    
    section_patterns = [
        r'\n(?:Section|SECTION)\s+\d+[A-Z]?\.?\s*—',
        r'\n(?:Section|SECTION)\s+\d+[A-Z]?\.?\s+[A-Z]',
        r'\n\d+\.\s+[A-Z][a-z]+\s+[a-z]+\.—',
        r'\n(?:CHAPTER|Chapter)\s+[IVX]+\n',
        r'\n(?:CHAPTER|Chapter)\s+\d+\n',
        r'\n\([a-z0-9]+\)\s+',
        r'\n(?:Provided|Explanation)\.—',
    ]
    
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    
    current_chunk = ""
    current_section = ""
    current_chapter = ""
    
    for para in paragraphs:
        is_section = any(re.search(pattern, '\n' + para) for pattern in section_patterns[:3])
        is_chapter = any(re.search(pattern, '\n' + para) for pattern in section_patterns[3:5])
        
        if is_chapter:
            current_chapter = para[:100]
        
        if is_section:
            current_section = para[:150]
        
        potential_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
        
        if len(potential_chunk) >= chunk_size:
            if current_chunk:
                metadata_info = extract_section_info(current_chunk)
                chunks.append({
                    'text': current_chunk,
                    'section': current_section,
                    'chapter': current_chapter,
                    'sections': metadata_info['sections'],
                    'has_definition': 'means' in current_chunk.lower() or 'definition' in current_chunk.lower()
                })
            
            words = current_chunk.split()
            if len(words) > 30:
                overlap_words = words[-30:]
                overlap_text = ' '.join(overlap_words)
            else:
                overlap_text = current_chunk
            
            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
        else:
            current_chunk = potential_chunk
    
    if current_chunk and len(current_chunk) > 100:
        metadata_info = extract_section_info(current_chunk)
        chunks.append({
            'text': current_chunk,
            'section': current_section,
            'chapter': current_chapter,
            'sections': metadata_info['sections'],
            'has_definition': 'means' in current_chunk.lower() or 'definition' in current_chunk.lower()
        })
    
    return chunks


def load_pdfs_from_folder(folder_path="./data/acts"):
    """Load all PDFs from folder and add to ChromaDB"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Creating data folder: {folder_path}")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"WARNING: Please add PDF files to {folder_path}")
        return 0
    
    try:
        existing_count = collection.count()
        if existing_count > 0:
            print(f"Database already has {existing_count} chunks loaded")
            print("Skipping PDF loading (delete chroma_db folder to reload)")
            return existing_count
    except:
        pass
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"WARNING: No PDF files found in {folder_path}")
        return 0
    
    documents = []
    metadatas = []
    ids = []
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                print(f"  Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
            
            chunks = smart_chunk_text(text, pdf_file.stem, chunk_size=800, overlap=200)
            
            for i, chunk_data in enumerate(chunks):
                if chunk_data['text'].strip():
                    documents.append(chunk_data['text'])
                    metadatas.append({
                        "source": pdf_file.stem,
                        "chunk_id": i,
                        "file_name": pdf_file.name,
                        "section": chunk_data.get('section', '')[:200],
                        "chapter": chunk_data.get('chapter', '')[:100],
                        "sections": ','.join(chunk_data.get('sections', []))[:100],
                        "has_definition": chunk_data.get('has_definition', False)
                    })
                    ids.append(f"{pdf_file.stem}_chunk_{i}")
            
            print(f"  Created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
    
    if documents:
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"\nTotal: {len(documents)} chunks added to database")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
    
    return len(documents)


def retrieve_context(query, top_k=3):
    """Retrieve relevant chunks with enhanced metadata"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "", []
        
        context_parts = []
        sources = set()
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            source = metadata.get('source', 'Unknown')
            section = metadata.get('section', '')
            sections = metadata.get('sections', '')
            
            if sections:
                header = f"[{source} - Sections: {sections}]"
            elif section:
                header = f"[{source} - {section}]"
            else:
                header = f"[{source}]"
            
            context_parts.append(f"{header}\n{doc}")
            sources.add(source)
        
        context = "\n\n---\n\n".join(context_parts)
        return context, list(sources)
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "", []


def call_claude_with_retry(prompt, system_prompt, max_tokens=800, temperature=0.2):
    """Call Claude API with silent retry logic"""
    retries = 0
    max_retries = 5
    delay = 3
    
    while retries < max_retries:
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text, None
            
        except RateLimitError as e:
            retries += 1
            if retries >= max_retries:
                return None, "The system is busy. Please try again in a moment."
            
            wait_time = min(delay * (2 ** retries), 60)
            time.sleep(wait_time)
            
        except APIError as e:
            return None, f"API Error. Please check your configuration."
        
        except Exception as e:
            return None, f"An error occurred. Please try again."
    
    return None, "Please try again in a moment."


def chat_with_rag(message, history, translate_tamil=False):
    """Main chat function with RAG"""
    
    try:
        count = collection.count()
        if count == 0:
            return "No documents loaded. Please add PDF files to ./data/acts/ and restart."
    except:
        return "Database not initialized. Please check the setup."
    
    context, sources = retrieve_context(message, top_k=3)
    
    if not context:
        return "I couldn't find relevant information for this query. Please try rephrasing your question."
    
    user_prompt = f"""Context:

{context}

---

Question: {message}

Provide a SHORT answer (2-3 sentences) with precise citations including section numbers."""
    
    if translate_tamil:
        user_prompt += "\n\nIMPORTANT: Translate your entire answer to Tamil. Keep citations in English format [Source: Act Name, Section X] but translate everything else to Tamil."
    
    answer, error = call_claude_with_retry(
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=1200 if translate_tamil else 800,
        temperature=0.2
    )
    
    if error:
        return error
    
    if sources and answer:
        answer += f"\n\n**Sources:** {', '.join(sorted(sources))}"
    
    return answer


print("\n" + "="*60)
print("POCKET LAWYER - Indian Legal Assistant POC")
print("="*60)
print("\nInitializing database...")
num_docs = load_pdfs_from_folder()

if num_docs == 0:
    print("\nWARNING: No documents loaded!")
    print("Add PDF files to: ./data/acts/")
else:
    print(f"\nReady! Loaded {num_docs} document chunks")

print("\nStarting Gradio interface...")
print("="*60 + "\n")

custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
.header-text {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(css=custom_css, title="Pocket Lawyer", theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
        <div class="header-text">
            <h1 style="margin: 0; font-size: 2.5em;">Pocket Lawyer</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Indian Legal Information Assistant</p>
            <p style="margin: 10px 0 0 0; font-size: 0.8em;">By Gokul Sahar S</p>
        </div>
    """)
    
    gr.HTML("""
        <div style="
        background: rgba(255, 230, 160, 0.15);
        border-left: 4px solid #FFD166;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 24px;
        color: #FFE8A3;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        line-height: 1.5;
    ">
    <strong style="color: #FFD166;">Important Disclaimer:</strong>
    <span style="color: #FFF6D6;">
        This application provides general legal information only and is not a substitute for
        professional legal advice. Always consult a qualified lawyer for your specific legal situation.
    </span>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                height=550,
                show_label=False,
                avatar_images=(None, None),
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask your legal question (e.g., 'What is the Consumer Protection Act?')",
                    show_label=False,
                    scale=5,
                    container=False
                )
                submit = gr.Button("Send", variant="primary", scale=1, size="lg")
            
            with gr.Row():
                tamil_checkbox = gr.Checkbox(label="Respond in Tamil", value=False)
                clear = gr.Button("Clear Chat", variant="secondary", size="sm")
        
        with gr.Column(scale=3):
            gr.Markdown("### Example Questions")
            examples = gr.Examples(
                examples=[
                    "What is the Consumer Protection Act?",
                    "How do I file an RTI application?",
                    "What are consumer rights for defective products?"
                ],
                inputs=msg,
                label=""
            )
    
    def respond(message, chat_history, tamil_mode):
        if not message.strip():
            return "", chat_history
        
        bot_message = chat_with_rag(message, chat_history, tamil_mode)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    def clear_chat():
        return "", []
    
    msg.submit(respond, [msg, chatbot, tamil_checkbox], [msg, chatbot])
    submit.click(respond, [msg, chatbot, tamil_checkbox], [msg, chatbot])
    clear.click(clear_chat, None, [msg, chatbot])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )