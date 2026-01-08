import streamlit as st
from core_logic import ITAssistant
from PIL import Image

st.set_page_config(page_title="IT Support Assistant", page_icon="icons/it_support.png")

@st.cache_resource
def get_bot():
    return ITAssistant()

# --- SIDEBAR ---
with st.sidebar:
    st.image("icons/wakefield.png", width=200)
    st.title("About")
    st.markdown("""
        This assistant utilizes **Advanced Retrieval-Augmented Generation** to provide frontline technical support.
        
        **How it works:**
        1. **Initial Triage:** Your query is routed to a specific domain to troubleshoot (Hardware, Security, Networking, General).
        2. **KB Search:** The system queries the internal IT knowledge base for a documented solution that aligns with company policy.
        3. **Troubleshoot Instructions:** The AI agent generates simple instructions for you to follow as identified by your IT department.
        
        """)
    
    st.title("Core Tech Stack")
    st.info("""
        - **LLM:** Llama 3.2 (3B)
        - **Embeddings:** Nomic-Embed-Text
        - **Orchestration:** LlamaIndex
        - **Database:** Qdrant (Local)
    """)
    st.markdown("---")

    if st.button("Contact Help Desk"):
        st.toast("Redirecting to IT Portal...")


# Initialize engine and message history
if "bot" not in st.session_state:
    st.session_state.bot = get_bot()
if "messages" not in st.session_state:
    st.session_state.messages = []



# Heading 
logo = Image.open("icons/vito.png")
col1, col2 = st.columns([0.20, 0.80])
with col1:
    st.image(logo, width=120)  
with col2:
    st.title("IT Support Assistant")

    

# Display Chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    # Generate Response
    with st.chat_message("assistant"):
        # Categorize Issue 
        with st.spinner("Analyzing request..."):
            category = st.session_state.bot.triage_classify(prompt)
            st.caption(f"📍 Routed to: **{category.upper()}**")
        
        # Response
        with st.spinner("Searching internal knowledge..."):
            response = st.session_state.bot.run_query(prompt, category)
        
            placeholder = st.empty()
            full_response = ""
        
        # Stream the output text
            for token in response.response_gen:
                full_response += token
                placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        
        # Show Sources
        if response.source_nodes:
            with st.expander("Sources"):
                for node in response.source_nodes:
                    st.write(f"- {node.node.metadata.get('file_name')} (Match Score: {node.score:.2f})")

    # Session state history
    st.session_state.messages.append({"role": "assistant", "content": full_response})