import os
import streamlit as st
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


openai_api_key = "sk-proj-ZtaEi190d28t06TuBNjST3BlbkFJ6byccsPVXzCGuFIPiPJV"
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

# LOAD THE VECTOR DATABASE AND PREPARE RETRIEVAL
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma.db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)
llm = ChatOpenAI( model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.2)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory()

# CONTEXT PROMPT
### Contextualize question ###
contextualize_q_assistant_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("assistant", contextualize_q_assistant_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# PREPARE PROMPT
assistant_prompt = (
    #"You are a friendly clinician at the National Institutes of Health (NIH)."
    "Your task is to answer clients questions as truthfully as possible."
    "Use the provided retrieved information  "
    "help answer the questions. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("assistant", assistant_prompt), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# TITLE
st.markdown("<h1 style='text-align: center;'>Demo DY consultant</h1> <br>", unsafe_allow_html=True)

st.info(
    "Chat with  Federal Aviation Administration’s (FAA) standards and recommendations for airport design"
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if input := st.chat_input():
    st.chat_message("human").write(input)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "abc123"}}
    response = conversational_rag_chain.invoke({"input": input}, config)
    
    with st.chat_message("assistant"):
        st.write(response["answer"])
        with st.expander("See 3 first sources:"):
            for doc in response["context"]:
                source = os.path.split(doc.metadata["source"])[1] + "--->   Page: "+ str(doc.metadata["page"])
                st.write(source)

