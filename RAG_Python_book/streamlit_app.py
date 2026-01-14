import os
import re
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Byte of Python Chatbot",
    page_icon="🐍",
    layout="wide"
)

# -------- CONFIG --------
BOOK_PATH = os.getenv("BOOK_PATH", r"C:\Archana_Docs\Python_Chatbot\byte_of_python.txt")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 10000))
# ------------------------


def load_book(path):
    """Load the book content from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_chapters(book_text):
    """Parse the book into chapters based on common patterns."""
    chapters = {}
    
    chapter_patterns = [
        r'\n(Chapter\s+\d+[:\s]+[^\n]+)',
        r'\n(\d+\.\s+[A-Z][^\n]{10,80})\n',
        r'\n(#{1,3}\s+[A-Z][^\n]{10,80})\n',
    ]
    
    all_matches = []
    for pattern in chapter_patterns:
        matches = re.finditer(pattern, book_text, re.MULTILINE)
        for match in matches:
            all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    all_matches.sort(key=lambda x: x[0])
    
    if not all_matches:
        return {"Full Book": book_text}
    
    for i, (start, end, title) in enumerate(all_matches):
        chapter_start = start
        chapter_end = all_matches[i + 1][0] if i + 1 < len(all_matches) else len(book_text)
        
        chapter_content = book_text[chapter_start:chapter_end]
        title = re.sub(r'^#+\s*', '', title).strip()
        chapters[title] = chapter_content
    
    return chapters


def find_best_matching_chapters(chapters, question):
    """Score and rank chapters by relevance to the question."""
    question_lower = question.lower()
    stop_words = {'what', 'how', 'does', 'work', 'explain', 'tell', 'about', 'python', 'chapter'}
    keywords = [w for w in question_lower.split() if len(w) > 3 and w not in stop_words]
    
    concept_keywords = {
        'while': ['while', 'loop', 'iterate', 'iteration', 'repeat'],
        'for': ['for', 'loop', 'iterate', 'iteration', 'range'],
        'if': ['if', 'elif', 'else', 'condition', 'conditional', 'control'],
        'function': ['function', 'def', 'define', 'call', 'return'],
        'class': ['class', 'object', 'oop', 'method', 'self'],
        'list': ['list', 'array', 'sequence', 'append'],
        'dict': ['dict', 'dictionary', 'key', 'value', 'mapping'],
        'string': ['string', 'str', 'text', 'character'],
        'module': ['module', 'import', 'package', 'library'],
        'exception': ['exception', 'error', 'try', 'except', 'raise'],
        'file': ['file', 'open', 'read', 'write', 'close'],
    }
    
    expanded_keywords = set(keywords)
    for keyword in keywords:
        if keyword in concept_keywords:
            expanded_keywords.update(concept_keywords[keyword])
    
    chapter_scores = []
    for chapter_name, chapter_content in chapters.items():
        content_lower = chapter_content.lower()
        title_lower = chapter_name.lower()
        
        content_score = sum(content_lower.count(kw) for kw in expanded_keywords)
        title_score = sum(title_lower.count(kw) * 20 for kw in expanded_keywords)
        
        phrase_score = 0
        if len(keywords) > 1:
            phrase = ' '.join(keywords[:3])
            if phrase in content_lower:
                phrase_score = 50
        
        total_score = content_score + title_score + phrase_score
        
        if total_score > 0:
            chapter_scores.append((chapter_name, chapter_content, total_score))
    
    chapter_scores.sort(key=lambda x: x[2], reverse=True)
    return chapter_scores


def extract_relevant_sections(content, keywords, max_length=3000):
    """Extract the most relevant sections from content based on keyword density."""
    if len(content) <= max_length:
        return content
    
    paragraphs = content.split('\n\n')
    scored_paragraphs = []
    
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(para_lower.count(kw) for kw in keywords)
        if score > 0 or len(para) > 100:
            scored_paragraphs.append((para, score))
    
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    total_length = 0
    for para, score in scored_paragraphs:
        if total_length + len(para) <= max_length:
            result.append(para)
            total_length += len(para)
        else:
            break
    
    return '\n\n'.join(result)


def find_relevant_context(book_text, question, chapters=None):
    """Enhanced retrieval using chapter structure and intelligent matching."""
    if chapters is None:
        chapters = parse_chapters(book_text)
    
    question_lower = question.lower()
    
    is_chapter_query = 'chapter' in question_lower and any(
        word in question_lower for word in ['summarize', 'summary', 'explain', 'tell', 'about']
    )
    
    relevant_chapters = find_best_matching_chapters(chapters, question)
    
    if not relevant_chapters:
        return book_text[:MAX_CONTEXT_CHARS]
    
    keywords = [w for w in question_lower.split() if len(w) > 3]
    
    if is_chapter_query:
        context = ""
        for chapter_name, chapter_content, score in relevant_chapters[:2]:
            context += f"\n\n{'='*70}\nCHAPTER: {chapter_name}\n{'='*70}\n\n{chapter_content[:5000]}\n"
        return context[:MAX_CONTEXT_CHARS]
    else:
        context = ""
        for chapter_name, chapter_content, score in relevant_chapters[:3]:
            relevant_section = extract_relevant_sections(chapter_content, keywords, max_length=3500)
            context += f"\n\n{'='*70}\nFrom Chapter: {chapter_name}\n{'='*70}\n\n{relevant_section}\n"
            
            if len(context) >= MAX_CONTEXT_CHARS:
                break
        
        return context[:MAX_CONTEXT_CHARS]


def build_system_prompt():
    """Create the system prompt for the AI tutor."""
    return (
        "You are an expert Python tutor specializing in 'A Byte of Python' book.\n\n"
        "Guidelines:\n"
        "1. If the user asks for a chapter summary, provide a comprehensive overview "
        "of the chapter's main topics, concepts, and key takeaways.\n"
        "2. For specific questions (like 'how does while loop work'), provide:\n"
        "   - Clear explanation of the concept\n"
        "   - Practical, executable code example\n"
        "   - Common use cases or best practices\n"
        "3. Base your answer primarily on the provided book excerpt.\n"
        "4. If the excerpt doesn't contain complete information, use your general Python knowledge "
        "to supplement, but mention that you're doing so.\n"
        "5. Always include at least one working code example.\n"
        "6. Be encouraging and make concepts easy to understand."
    )


def ask_groq(client, context, question):
    """Send question with context to Groq API and get response."""
    system_prompt = build_system_prompt()
    
    is_summary = any(word in question.lower() for word in ['summarize', 'summary', 'overview'])
    is_chapter_specific = 'chapter' in question.lower()
    
    if is_summary and is_chapter_specific:
        user_message = (
            f"Question: {question}\n\n"
            f"Based on the following content from 'A Byte of Python', provide a comprehensive chapter summary "
            f"covering the main concepts, topics, and learning objectives:\n\n"
            f"{context}"
        )
    else:
        user_message = (
            f"Question: {question}\n\n"
            f"Here are the relevant sections from 'A Byte of Python':\n\n"
            f"{context}\n\n"
            f"Based on this content, provide a clear answer with practical code examples."
        )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
        max_tokens=2000,
    )
    
    return response.choices[0].message.content


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_loaded" not in st.session_state:
    st.session_state.book_loaded = False
    st.session_state.book_text = None
    st.session_state.chapters = None


# Sidebar
with st.sidebar:
    st.title("🐍 Byte of Python Chatbot")
    st.markdown("---")
    
    # Load book button
    if not st.session_state.book_loaded:
        if st.button("📚 Load Book", use_container_width=True):
            with st.spinner("Loading book and parsing chapters..."):
                try:
                    st.session_state.book_text = load_book(BOOK_PATH)
                    st.session_state.chapters = parse_chapters(st.session_state.book_text)
                    st.session_state.book_loaded = True
                    st.success(f"✓ Book loaded! Found {len(st.session_state.chapters)} chapters")
                except Exception as e:
                    st.error(f"Error loading book: {e}")
    else:
        st.success(f"✓ Book loaded ({len(st.session_state.chapters)} chapters)")
        if st.button("🔄 Reload Book", use_container_width=True):
            st.session_state.book_loaded = False
            st.rerun()
    
    st.markdown("---")
    
    # Example questions
    st.markdown("### 💡 Try asking:")
    st.markdown("""
    - Summarize chapter 5
    - How does while loop work?
    - What are functions?
    - Explain classes in Python
    - Tell me about lists
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Model: {MODEL_NAME}")


# Main chat interface
st.title("💬 Chat with Byte of Python")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Python..."):
    # Check if book is loaded
    if not st.session_state.book_loaded:
        st.error("⚠️ Please load the book first using the sidebar button!")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Finding relevant content..."):
            try:
                # Get API key
                api_key = os.getenv("Groq_public_Key")
                if not api_key:
                    st.error("❌ Groq API key not found in .env file!")
                    st.stop()
                
                # Initialize Groq client
                client = Groq(api_key=api_key)
                
                # Find relevant context
                context = find_relevant_context(
                    st.session_state.book_text, 
                    prompt, 
                    st.session_state.chapters
                )
                
                # Get response from Groq
                response = ask_groq(client, context, prompt)
                
                # Display response
                st.markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
