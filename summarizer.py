import os
import glob
import pdfplumber
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 4
CHROMA_PERSIST_DIR = "./chroma_legal_db"
PDF_FOLDER = "./legal_docs"  # Fixed folder for legal documents

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

class DocumentProcessor:
    """Handles PDF processing and chunking"""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_and_process_pdfs(pdf_folder: str) -> List[Document]:
        """Load and process all PDFs from the legal_docs folder"""
        documents = []
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        
        if not pdf_files:
            st.error(f"No PDF files found in {pdf_folder}")
            return documents
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for pdf_path in pdf_files:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    if text:
                        # Create document chunks
                        chunks = text_splitter.split_text(text)
                        for i, chunk in enumerate(chunks):
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": os.path.basename(pdf_path),
                                    "chunk_id": i
                                }
                            )
                            documents.append(doc)
            except Exception as e:
                st.error(f"Error processing {pdf_path}: {str(e)}")
                continue
        
        return documents

class RAGSystem:
    """Main RAG system with retrieve and rerank"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.qa_chain = None
    
    def initialize_vectorstore(self, documents: List[Document]):
        """Initialize or load vector store"""
        if os.path.exists(CHROMA_PERSIST_DIR):
            # Load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings
            )
        else:
            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            self.vectorstore.persist()
        
        return self.vectorstore
    
    def create_qa_chain(self):
        """Create QA chain with reranking"""
        if not OPENAI_API_KEY:
            st.error("⚠️ OpenAI API key not found. Please add it to your .env file")
            return None
        
        # Initialize LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            model_name="gpt-3.5-turbo"
        )
        
        # Base retriever
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )
        
        # Cross-encoder for reranking
        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        compressor = CrossEncoderReranker(
            model=model,
            top_n=TOP_K_RERANK
        )
        
        # Compression retriever with reranking
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Custom prompt for legal context
        prompt_template = """You are an expert legal assistant specialized in analyzing Indian legal documents.
        Your role is to provide accurate, helpful answers based on the legal documents provided.
        
        Context from legal documents:
        {context}
        
        Human Question: {question}
        
        Instructions:
        - If asked to summarize, provide a clear and concise summary of the relevant legal provisions
        - For questions, provide detailed answers with specific references to the legal documents
        - Always cite the source document when providing information
        - If the answer is not in the provided context, say so clearly
        - Use professional legal language while keeping explanations clear
        
        Assistant Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain

@st.cache_resource(show_spinner="🔄 Loading legal documents...")
def initialize_system():
    """Initialize the complete RAG system"""
    # Load and process documents
    processor = DocumentProcessor()
    documents = processor.load_and_process_pdfs(PDF_FOLDER)
    
    if not documents:
        return None, None
    
    # Initialize RAG system
    rag_system = RAGSystem()
    vectorstore = rag_system.initialize_vectorstore(documents)
    qa_chain = rag_system.create_qa_chain()
    
    return vectorstore, qa_chain

def format_sources(source_documents):
    """Format source documents for display"""
    sources = {}
    for doc in source_documents:
        source_name = doc.metadata.get("source", "Unknown")
        if source_name not in sources:
            sources[source_name] = []
        sources[source_name].append(doc.page_content[:200] + "...")
    return sources

def main():
    # Custom CSS for chat interface
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-content {
        padding: 0.5rem;
    }
    .sources-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
    }
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚖️ Legal Document Assistant")
    st.markdown("Ask questions or request summaries about Indian legal documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Initializing legal document system..."):
            vectorstore, qa_chain = initialize_system()
            
            if vectorstore and qa_chain:
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = qa_chain
                st.session_state.initialized = True
                st.success("✅ System ready! You can now ask questions about the legal documents.")
            else:
                st.error("Failed to initialize system. Please check if legal documents are present in the legal_docs folder.")
                return
    
    # Sidebar with options
    with st.sidebar:
        st.header("📋 Options")
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("📝 Summarize All Documents", use_container_width=True):
            st.session_state.next_message = "Please provide a comprehensive summary of all legal documents"
        
        if st.button("🔍 List Key Provisions", use_container_width=True):
            st.session_state.next_message = "What are the key legal provisions in these documents?"
        
        if st.button("📊 Extract Important Sections", use_container_width=True):
            st.session_state.next_message = "Extract and list all important sections from the legal documents"
        
        st.divider()
        
        # Chat history management
        st.subheader("Chat History")
        st.write(f"Total messages: {len(st.session_state.chat_history)}")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Export Chat History", use_container_width=True):
            if st.session_state.chat_history:
                chat_text = "\n\n".join([
                    f"Q: {item['question']}\nA: {item['answer']}\nTime: {item['timestamp']}\n"
                    for item in st.session_state.chat_history
                ])
                st.download_button(
                    label="Download Chat History",
                    data=chat_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        st.divider()
        
        # System info
        st.subheader("ℹ️ System Info")
        st.info("""
        Documents loaded from: `legal_docs/`
        - indian_laws1.pdf
        - indian_laws2.pdf
        
        Features:
        - Semantic search
        - Cross-encoder reranking
        - GPT-3.5 responses
        """)
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-content">
                <strong>You:</strong> {message['question']}
            </div>
            <div class="timestamp">{message['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-content">
                <strong>Legal Assistant:</strong> {message['answer']}
            </div>
        """, unsafe_allow_html=True)
        
        # Sources
        if message.get('sources'):
            sources_html = "<div class='sources-box'><strong>📚 Sources:</strong><ul>"
            for source, excerpts in message['sources'].items():
                sources_html += f"<li>{source}</li>"
            sources_html += "</ul></div>"
            st.markdown(sources_html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Check for quick action message
    if hasattr(st.session_state, 'next_message'):
        question = st.session_state.next_message
        del st.session_state.next_message
    else:
        question = None
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_area(
                "Your question:",
                value=question if question else "",
                placeholder="Ask about legal provisions, request summaries, or inquire about specific sections...",
                height=100,
                key="user_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(
                "Send 📤",
                use_container_width=True,
                type="primary"
            )
    
    # Process user input
    if submit_button and user_input:
        with st.spinner("🔍 Searching legal documents and generating response..."):
            try:
                # Get response from QA chain
                response = st.session_state.qa_chain({"query": user_input})
                
                answer = response.get("result", "I couldn't find an answer to your question.")
                source_documents = response.get("source_documents", [])
                
                # Format sources
                sources = format_sources(source_documents) if source_documents else None
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": answer,
                    "sources": sources,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Rerun to display new message
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>💡 Tip: Use the sidebar for quick actions and to manage your chat history</small></center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
