import streamlit as st
from dotenv import load_dotenv
import asyncio
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import pickle
import dill
import io

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# set the initial page configeration
st.set_page_config(
    page_title="PDF Chat", layout="wide", initial_sidebar_state="expanded"
)

# create embeddings for the chunks, set as global object
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

async def main():
    async def storeEmbedding(file, filename):
        '''
        :param file: pdf file to be parsed for creating embedding
        :param filename: .pkl filename where the vectors are to be stored
        :return: None
        '''
        reader = PdfReader(file)
        corpus = "".join([p.extract_text() for p in reader.pages if p.extract_text()])

        # split the corpus into chunk using langchain.text_splitter class method
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(corpus)

        # create embeddings for the chunks
        # embeddings = OpenAIEmbeddings(openai_api_key=api_key) # already set as global object
        # store embeddings for the chunks in vector database
        vectors = FAISS.from_texts(chunks, embeddings)

        # store vectors in pickle file
        # we are storing the vectors in pickle file so that if user comes again to chatbot and uploads same file again, we dont need to create embedding vectors again for the same file
        vectors.save_local(filename)
        #with open(filename + ".pkl", "wb") as f:
        #    pickle.dump(vectors, f)

    async def getDocsEmbed(file, filename):
        '''
        :param file: pdf file to be parsed for creating embedding
        :param filename: pkl file where embedding were stored
        :return:
        '''
        # check if .pkl file exists
        #if not os.path.isfile(filename + '.pkl'):
        #    await storeEmbedding(file, filename)
        if not os.path.isfile(filename):
            await storeEmbedding(file, filename)

        #with open(filename + '.pkl', "rb") as f:
        #    vectors = pickle.load(f)
        vectors = FAISS.load_local(filename, embeddings)

        return vectors

    async def conversational_chat(query):
        result = qa({
            "question": query,
            "chat_history": st.session_state["history"]
        })
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]


    # overall history for the bot
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.title("PDF Chat")

    option = st.selectbox("Select Option", ("PDF", "Blog", "DB"))

    if option == "PDF":
        uploaded_file = st.file_uploader("Choose a pdf file", type="PDF")
        if uploaded_file is not None:
            with st.spinner("Processing ..."):
                uploaded_file.seek(0)
                # read the entire file
                file = uploaded_file.read()
                vectors = await getDocsEmbed(io.BytesIO(file), uploaded_file.name)
                # now, we get the vectors, we would like to start conversational retreival chain using langchain
                # we want to retreive the previous conversations and pass it to the next prompt so that the LLM models
                # can understand context better, so lets
                # initialize conversational retreival chain
                qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo"),
                                                           retriever=vectors.as_retriever(),
                                                           return_source_documents=True)


            # set a boolean variable to see if user has uploaded something or not
            st.session_state["ready"] = True

    elif option == "Blog":
        url = st.text_input("Enter the url of the blog")

        if url:
            st.session_state["ready"] = True

    elif option == "DB":
        uploaded_file = st.file_uploader("Choose a Database file", type="db")
        if uploaded_file is not None:
            with st.spinner("Processing ..."):
                uploaded_file.seek(0)

            st.session_state["ready"] = True

    # check if user has uploaded something by looking at boolean variable in the dict
    if st.session_state.get("ready", False):
        # generated key correspond to responses generated
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["Welcome! you can ask any questions!"]

        # past key corresponds to what is asked by user
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey"]

        # create a container for user to ask questions
        container = st.container()
        # create a container for the chatbot to render response
        response_container = st.container()

        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Query:", placeholder="e.g., Summarize the document", key="input"
                )
                submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input:
                # this is the response from LLM
                output = await conversational_chat(user_input)
                # update the session_state key values
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

        # check if our chatbot has generated some output
        if st.session_state["generated"]:
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    if i < len(st.session_state["past"]):
                        # create a markdown, note right aligned in markdown
                        st.markdown(
                            "<div style='background-color: #90caf9; color: black; padding: 10px; border-radius: 5px; width: 70%; float: right; margin: 5px;'>"
                            + st.session_state["past"][i]
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                        # show generated response in streamlit-chat message
                        message(
                            st.session_state["generated"][i],
                            key=str(i),
                            avatar_style="fun-emoji",
                        )


if __name__ == "__main__":
    asyncio.run(main())
