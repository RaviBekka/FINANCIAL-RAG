
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Page config
st.set_page_config(
    page_title="RIL Financial Analyst",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Reliance Industries Financial Analyst")
st.caption("Ask questions about RIL Annual Reports 2022-23 and 2023-24")

# Load vector store
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        "/content/ril_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# Load LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key="Your-key",
        temperature=0
    )

vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst assistant for Reliance Industries.
Use ONLY the context below to answer questions.
If the answer is not in the context, say 'I couldn't find this in the reports.'
Always mention which report year your answer is from.

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join([
        f"[{doc.metadata['source']}, Page {doc.metadata.get('page','N/A')}]\n{doc.page_content}"
        for doc in docs
    ])

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask about Reliance Industries..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching reports..."):
            docs = retriever.invoke(question)
            context = format_docs(docs)

            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "chat_history": st.session_state.chat_history,
                "question": question
            })

            answer = response.content
            st.markdown(answer)

            # Show sources
            with st.expander("📄 Sources"):
                for doc in docs[:3]:
                    st.caption(f"• {doc.metadata['source']}, Page {doc.metadata.get('page','N/A')}")

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
