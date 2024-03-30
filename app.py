import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db =  FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    # Set page configuration and theme
    st.set_page_config(page_title="Chat PDF", page_icon="💁", layout="wide", initial_sidebar_state="expanded")

    # Custom CSS to style the sidebar and button
    sidebar_css = """
    <style>
    .st-emotion-cache-6qob1r {
    background-color: #0B233F;
}
.st-emotion-cache-taue2i {
    background-color: #015A68;

    color: rgb(49, 51, 63);
}
.st-emotion-cache-taue2i {
    color: rgb(49, 51, 63);
}
.st-emotion-cache-9ycgxx {
    margin-bottom: 0.25rem;
    color: white;
    margin-top:20px
}
st-emotion-cache-16txtl3 h1 {
    color: white;
}
.st-emotion-cache-l9bjmx p {
    font-size: 14px;
    color: white;
    margin-top:20px
}
.st-emotion-cache-16txtl3 h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: white;
    margin-top:20px

}
.st-emotion-cache-7ym5gk {
    color: white;
    background-color: rgb(49, 51, 63);
    border: 1px solid rgba(49, 51, 63, 0.2);
}
.st-emotion-cache-10trblm {
    color: #0B233F;
}
.st-emotion-cache-1pbsqtx {
    color: white;
}
.st-emotion-cache-13ejsyy {
   
    color: white;
    background-color: black;
    border: 1px solid rgba(49, 51, 63, 0.2);
}
.st-emotion-cache-1aehpvj {
    color: white;
    
}
.st-bv {
    caret-color: rgb(49, 51, 63);
    padding: 14px;
}
input .st-emotion-cache-l9bjmx p {
    font-size: 14px;
    color: #0b233f;
    margin-top: 20px;
}
.st-emotion-cache-1uixxvy {
    color: white;
}

    </style>
    """
    st.markdown(sidebar_css, unsafe_allow_html=True)

    # Define the main content
    st.header("LinguaGlimmer: Illuminating Insights from Pdfs")
    st.text('This innovative tool empowers users to engage in dynamic conversations with multiple PDF documents effortlessly.')

    user_question = st.text_input("",placeholder="Type your question here...")

    if st.button("Submit", key="submit_button"):
        user_input(user_question)

    with st.sidebar:
        # Custom CSS to style the file uploader button
        file_uploader_css = """
        <style>
        .st-bm button {
            background-color: purple !important;
            color: white !important;
        }
        </style>
        """
        st.markdown(file_uploader_css, unsafe_allow_html=True)

        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process", key="submit_process_button"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
