import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
import faiss
import numpy as np
from openai import OpenAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Govt AI Assistant",
    layout="wide"
)

# -----------------------------
# OPENAI SETUP
# -----------------------------
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["OPENAI_API_KEY"]
    except:
        API_KEY = None

if not API_KEY:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# -----------------------------
# APP HEADER
# -----------------------------
st.title("🏛️ Govt AI Assistant (MVP)")
st.write(
    "AI-powered assistant for Governance, Audit, Health & Panchayat use-cases"
)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def ask_llm(
    prompt,
    system_prompt="You are a helpful government assistant."
):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}"


def extract_pdf_text(uploaded_file):

    try:
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""


def chunk_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_vector_store(chunks):

    embeddings = []

    for chunk in chunks:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        embeddings.append(
            response.data[0].embedding
        )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_document(query):

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    query_embedding = np.array(
        [query_embedding],
        dtype=np.float32
    )

    index = st.session_state.index
    chunks = st.session_state.chunks

    k = min(3, len(chunks))

    D, I = index.search(query_embedding, k)

    results = []

    for idx in I[0]:
        results.append(chunks[idx])

    return "\n\n".join(results)


# -----------------------------
# SIDEBAR
# -----------------------------
option = st.sidebar.selectbox(
    "Select Module",
    [
        "📄 Document Q&A",
        "📝 Memo Generator",
        "🤖 General Chatbot",
        "💰 Audit Compliance Checker",
        "🏥 Health Citizen Chatbot"
    ]
)

# -----------------------------
# DOCUMENT Q&A
# -----------------------------
if option == "📄 Document Q&A":

    st.header("📄 Ask Questions from Policy Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        file_name = uploaded_file.name

        if (
            "current_pdf" not in st.session_state
            or st.session_state.current_pdf != file_name
        ):

            with st.spinner("Reading PDF..."):

                text = extract_pdf_text(uploaded_file)

            st.info(
                f"Characters Extracted: {len(text)}"
            )

            if len(text.strip()) == 0:

                st.error(
                    "No text found in PDF. "
                    "This may be a scanned PDF."
                )

            else:

                chunks = chunk_text(text)

                st.info(
                    f"Chunks Created: {len(chunks)}"
                )

                with st.spinner(
                    "Creating embeddings..."
                ):

                    index = create_vector_store(chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.current_pdf = file_name

                st.success(
                    "Document Ready for Q&A"
                )

        if "index" in st.session_state:

            query = st.text_input(
                "Ask your question"
            )

            if query:

                with st.spinner(
                    "Searching document..."
                ):

                    context = search_document(query)

                    prompt = f"""
Answer ONLY using the document context.

If the answer is unavailable,
reply:

The information is not available in the uploaded document.

DOCUMENT:
{context}

QUESTION:
{query}
"""

                    answer = ask_llm(prompt)

                st.markdown("### Answer")
                st.success(answer)

# -----------------------------
# MEMO GENERATOR
# -----------------------------
elif option == "📝 Memo Generator":

    st.header("📝 Government Memo Generator")

    issue = st.text_area("Enter Issue")

    if st.button("Generate Memo"):

        if issue:

            prompt = f"""
Draft an official government memo.

Issue:
{issue}

Format:

Subject

Background

Observations

Action Required

Signature
"""

            result = ask_llm(prompt)

            st.markdown(result)

        else:
            st.warning(
                "Please enter an issue."
            )

# -----------------------------
# GENERAL CHATBOT
# -----------------------------
elif option == "🤖 General Chatbot":

    st.header("🤖 Citizen Helpdesk")

    user_query = st.text_input(
        "Ask your question"
    )

    if user_query:

        result = ask_llm(user_query)

        st.success(result)

# -----------------------------
# AUDIT COMPLIANCE
# -----------------------------
elif option == "💰 Audit Compliance Checker":

    st.header(
        "💰 Audit Compliance Checker"
    )

    uploaded_csv = st.file_uploader(
        "Upload Expense File",
        type=["csv"]
    )

    if uploaded_csv:

        df = pd.read_csv(uploaded_csv)

        st.dataframe(df.head())

        if st.button("Check Compliance"):

            prompt = f"""
Analyze the expense records.

{df.head(30).to_string()}

Identify:

1. Policy violations
2. Suspicious spending
3. Duplicate expenses
4. Budget risks
5. Recommendations
"""

            result = ask_llm(prompt)

            st.markdown(result)

# -----------------------------
# HEALTH CHATBOT
# -----------------------------
elif option == "🏥 Health Citizen Chatbot":

    st.header(
        "🏥 Health Department Chatbot"
    )

    query = st.text_input(
        "Enter your health query"
    )

    if query:

        result = ask_llm(
            query,
            system_prompt="""
You are a Government Health Officer.

Provide information based on
Indian public health guidelines.

Do not provide medical diagnosis.
"""
        )

        st.success(result)
