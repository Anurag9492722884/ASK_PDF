import tempfile
from PIL import Image


import os

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

import streamlit as st


from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma


from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


# theme = {
#     "backgroundColor": "rgb(77 143 217)",
# } 
# st.set_page_config(
#     **theme
# )
st.title('🦜🔗 PDF-Chat: Interact with Your PDFs in a Conversational Way')
st.subheader('Load your PDF, ask questions, and receive answers directly from the document.')


image = Image.open('langchain-chat-with-pdf.png')
st.image(image)

st.subheader('Upload your pdf')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))

temp_file_path = os.getcwd()
while uploaded_file is None:
    x = 1
        
if uploaded_file is not None:
    
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)


os.environ['OPENAI_API_KEY'] = "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"

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()


loader = PyPDFLoader(temp_file_path)

pages = loader.load_and_split()


store = Chroma.from_documents(pages, embeddings, collection_name='collectio')

vectorstore_info = VectorStoreInfo(
    name="collectio",
    description=" A pdf file to answer your questions",
    vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

prompt = st.text_input('Input your prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt) 
        st.write(search[0][0].page_content)
