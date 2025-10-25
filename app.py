"""
Pocket Lawyer - Indian Legal Assistant
Simple RAG chatbot for legal questions with citations
"""

import os
import gradio as gr
from anthropic import Anthropic
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize ChromaDB (local, file-based)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # Small, fast, free
)

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="legal_docs",
    embedding_function=sentence_transformer_ef
)

# System prompt for Claude
SYSTEM_PROMPT = """You are a helpful legal information assistant for Indian citizens. 

CRITICAL RULES:
1. Only answer using the provided context documents
2. Always cite sources in this format: [Act Name, Section X]
3. Use simple, plain English - avoid legal jargon
4. If you don't find relevant information in context, say "I don't have enough information in my knowledge base to answer this accurately."
5. Add this disclaimer at the end: " This is general information only, not legal advice. Consult a lawyer for your specific situation."

TONE: Helpful, clear, and citizen-friendly"""


def load_pdfs_from_folder(folder_path="./data/acts"):
    """Load all PDFs from folder and add to ChromaDB"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Creating data folder: {folder_path}")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  Please add PDF files to {folder_path}")
        return 0
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"  No PDF files found in {folder_path}")
        return 0
    
    documents = []
    metadatas = []
    ids = []
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            
            # Extract text from PDF
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + "\n"
            
            # Simple chunking (split by paragraphs, ~500 chars each)
            chunks = []
            current_chunk = ""
            
            for paragraph in text.split('\n\n'):
                if len(current_chunk) + len(paragraph) < 500:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Add chunks to collection
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    documents.append(chunk)
                    metadatas.append({
                        "source": pdf_file.stem,
                        "chunk_id": i,
                        "file_name": pdf_file.name
                    })
                    ids.append(f"{pdf_file.stem}_chunk_{i}")
            
            print(f"✓ Loaded {len(chunks)} chunks from {pdf_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {e}")
    
    # Add all documents to ChromaDB
    if documents:
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"\n✓ Total: {len(documents)} chunks added to database")
        except Exception as e:
            print(f"✗ Error adding to ChromaDB: {e}")
    
    return len(documents)


def retrieve_context(query, top_k=5):
    """Retrieve relevant chunks from ChromaDB"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "", []
        
        # Format context with sources
        context_parts = []
        sources = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            source = metadata.get('source', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{doc}")
            if source not in sources:
                sources.append(source)
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "", []


def chat_with_rag(message, history):
    """Main chat function with RAG"""
    
    # Check if database is empty
    try:
        count = collection.count()
        if count == 0:
            return " No documents loaded yet. Please add PDF files to the ./data/acts folder and restart the app."
    except:
        return " Database not initialized. Please check the setup."
    
    # Retrieve relevant context
    context, sources = retrieve_context(message, top_k=5)
    
    if not context:
        return "I couldn't find relevant information in my knowledge base for this query. Please try rephrasing your question or ask about topics covered in the loaded documents."
    
    # Build prompt for Claude
    user_prompt = f"""Context from legal documents:

{context}

---

User Question: {message}

Please answer the question using ONLY the information from the context above. Include citations to the source documents."""
    
    # Call Claude API
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.content[0].text
        
        # Add sources at the end
        if sources:
            answer += f"\n\n **Sources consulted:** {', '.join(sources)}"
        
        return answer
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your ANTHROPIC_API_KEY in the .env file."


# Initialize database on startup
print("\n" + "="*60)
print("🏛️  POCKET LAWYER - Indian Legal Assistant POC")
print("="*60)
print("\nInitializing database...")
num_docs = load_pdfs_from_folder()

if num_docs == 0:
    print("\n  WARNING: No documents loaded!")
    print(" Add PDF files to: ./data/acts/")
    print("   Then restart the app.\n")
else:
    print(f"\n✓ Ready! Loaded {num_docs} document chunks")

print("\nStarting Gradio interface...")
print("="*60 + "\n")

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Pocket Lawyer") as demo:
    
    gr.Markdown("""
    # 🏛️ Pocket Lawyer - Indian Legal Assistant
    
    Ask questions about Indian legal matters in plain English. Get answers with citations from legal documents.
    
     **Disclaimer:** This provides general information only, not legal advice. Consult a qualified lawyer for your specific situation.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(None, "⚖️")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a legal question (e.g., 'How do I file an RTI application?')",
                    show_label=False,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Questions")
            examples = gr.Examples(
                examples=[
                    "What is the Consumer Protection Act?",
                    "How do I file an RTI application?",
                    "What are my rights as a tenant?",
                    "Can I get a refund for a defective product?",
                    "What is the time limit for filing a consumer complaint?",
                ],
                inputs=msg
            )
            
            gr.Markdown(f"""
            ###  Database Status
            **Documents loaded:** {num_docs} chunks
            
            ###  Setup
            1. Add PDFs to `./data/acts/`
            2. Set `ANTHROPIC_API_KEY` in `.env`
            3. Run: `python app.py`
            """)
    
    # Handle chat
    def respond(message, chat_history):
        bot_message = chat_with_rag(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Local only
        server_port=7860,
        share=False  # Set to True to get public link for demo
    )