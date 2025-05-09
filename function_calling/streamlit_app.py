import streamlit as st
import os
import sys
import base64
import json
from io import StringIO
import pandas as pd

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to Python path
sys.path.insert(0, project_root)

# Now import from the function_calling package
from function_calling.mcp_client_script import ETLAgent
import json

# Set page config
st.set_page_config(
    page_title="Agentic Data Transform",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state for chat history and uploaded files
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "etl_agent" not in st.session_state:
    st.session_state.etl_agent = ETLAgent()

# Custom CSS for better chat interface
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #f8fafc 0%, #e3e9f3 100%);
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        font-size: 1.1em;
        padding: 0.5em 1em;
    }
    .stButton>button {
        background: linear-gradient(90deg, #205081 0%, #2d9cdb 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1em;
        padding: 0.6em 2em;
        margin-top: 0.5em;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2d9cdb 0%, #205081 100%);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.7rem;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 8px rgba(32,80,129,0.07);
    }
    .chat-message.user {
        background: linear-gradient(90deg, #205081 0%, #2d9cdb 100%);
        color: white;
        align-items: flex-end;
    }
    .chat-message.assistant {
        background: #f8fafc;
        border: 1px solid #e3e9f3;
        color: #1a202c;
        align-items: flex-start;
    }
    .chat-message .content {
        display: flex;
        flex-direction: column;
    }
    .file-info {
        font-size: 0.85em;
        color: #205081;
        margin-top: 0.5rem;
    }
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e3e9f3;
        background: #fff;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #205081;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🔄 Agentic AI Data Transformation (ETL) with Amazon Bedrock and Athena")
st.markdown("""Transform your data using AWS generative AI and analytics""")

# Remove sidebar transformation prompt
# Add file upload section in sidebar (keep as is)
with st.sidebar:
    st.header("Upload Files")
    uploaded_file = st.file_uploader("Upload a file to analyze", type=['csv', 'json', 'txt', 'parquet', 'application/vnd.ms-excel'])
    if uploaded_file is not None:
        # Read file content based on file type
        try:
            if uploaded_file.type == 'text/csv':
                df = pd.read_csv(uploaded_file)
                file_content = df.to_csv(index=False)
            elif uploaded_file.type == 'application/json':
                df = pd.read_json(uploaded_file)
                file_content = df.to_json(orient='records')
            elif uploaded_file.type == 'application/vnd.ms-excel':
                df = pd.read_excel(uploaded_file)
                file_content = df.to_csv(index=False)
            else:
                file_content = uploaded_file.read().decode('utf-8')
            # Store file info in session state
            file_info = {
                'name': uploaded_file.name,
                'type': uploaded_file.type,
                'content': file_content
            }
            st.session_state.uploaded_files[uploaded_file.name] = file_info
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            # Display file preview
            st.subheader("File Preview")
            try:
                if uploaded_file.type in ['text/csv', 'application/vnd.ms-excel']:
                    st.dataframe(df.head())
                elif uploaded_file.type == 'application/json':
                    st.json(df.head().to_dict(orient='records'))
                else:
                    st.text(file_content[:500] + "..." if len(file_content) > 500 else file_content)
            except Exception as e:
                st.error(f"Error previewing file: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Main chat area
st.header("Chat with the ETL Agent")

# Display chat messages (with native tables for preview/result)
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="content">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Display file attachments if any
        if 'files' in message:
            for file_info in message['files']:
                st.markdown(f"""
                <div class="file-info">
                    📎 Attached: {file_info['name']} ({file_info['type']})
                </div>
                """, unsafe_allow_html=True)
        # Display preview/result tables if present (assistant messages only)
        if message.get('role') == 'assistant':
            if 'preview_table' in message:
                preview = message['preview_table']
                if preview['rows']:
                    st.dataframe(pd.DataFrame(preview['rows'], columns=preview['columns']), use_container_width=True)
                else:
                    st.markdown('empty')
            if 'result_table' in message:
                result = message['result_table']
                if result['rows']:
                    st.dataframe(pd.DataFrame(result['rows'], columns=result['columns']), use_container_width=True)
                else:
                    st.markdown('empty')

# Chat input (main area, always visible)
user_input = st.text_area("Type your request:", "", key="chat_input", height=80)
submit = st.button("Send")

# Only allow one file for now
file_info = list(st.session_state.uploaded_files.values())[0] if st.session_state.uploaded_files else None

if submit and user_input.strip() and file_info:
    # Append user message to chat history
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'files': [file_info]
    })
    with st.spinner("Processing transformation..."):
        etl_result = st.session_state.etl_agent.process_user_request(user_input, [file_info])
    # Append assistant message to chat history
    if etl_result and etl_result.get('status') == 'success':
        # Compose assistant message with schema and query (no HTML tables)
        assistant_content = f"""
**Table Preview (First 5 Rows)**

"""  # Table will be rendered below
        assistant_content += f"**Table Schema**\n\n```json\n{json.dumps(etl_result['schema'], indent=2)}\n```"
        assistant_content += f"**Transformation Query**\n\n```sql\n{etl_result['transformation_query']}\n```"
        assistant_content += f"**Transformation Result**\n\n"  # Table will be rendered below
        st.session_state.messages.append({
            'role': 'assistant',
            'content': assistant_content,
            'preview_table': etl_result['preview'],
            'result_table': etl_result['result']
        })
    elif etl_result and etl_result.get('status') == 'error':
        st.session_state.messages.append({
            'role': 'assistant',
            'content': f"<span style='color:red'>Error: {etl_result.get('message')}</span>"
        })
    elif etl_result and etl_result.get('status') == 'cleanup':
        st.session_state.messages.append({
            'role': 'assistant',
            'content': f"<span style='color:green'>{etl_result.get('message')}</span>"
        })
    st.rerun()

# Add a sidebar with information
with st.sidebar:
    st.header("About this AWS Sample")
    st.markdown("""Powered by Amazon Bedrock (Anthropic Claude) and Amazon Athena\
    \- Upload a file, then chat to transform your data using natural language\
    \- Athena SQL is generated and executed automatically\
    \- Results and previews are shown instantly\
    \
    **Supported file types:** CSV, JSON, Excel
    \
    **Sample questions:**
    - "Show me the available columns"
    - "Filter rows where status = 'active'"
    - "Group by region and sum sales"
    \
    This AWS code sample demonstrates how to combine generative AI and analytics for intelligent data transformation.
    """)
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.uploaded_files = {}
        st.session_state.chat_input = ""
        st.rerun() 