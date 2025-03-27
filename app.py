import streamlit as st
import random
import time
from google import genai
from google.genai import types
from utils.retriever import *
import os
import torch
torch.classes.__path__ = [] # add this line to manually set it to empty to prevent 'RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_'


def delete_uploaded_files():
    for file_name in os.listdir("uploaded_files"):
        file_path = os.path.join("uploaded_files", file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        

st.title("PDF chatbot")


uploaded_files = st.file_uploader(
    label="Upload PDF files you want to chat with",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    
    #If the uploaded_files directory does not exist, create it
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")

    #Save the uploaded files in the specified directory so that the simple directory loader can be used later
    for uploaded_file in uploaded_files:
        with open(f"uploaded_files/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
    

    #Create a vectorstore retriever object which takes an user input and returns the 10 most relevant snippets from the uploaded PDFs which are relevant to the question.
    vectorstore_retriever_object= create_vectorstore_retriever(directory="./uploaded_files")
    
    #Remove the uploaded files once the vectorstore retriever object has been created
    delete_uploaded_files()



    


    
    


    system_instruction="""You are a highly analytical, data-focused assistant.  
                        You are capable of critically analyzing any query given to you and providing the user with a helpful answer. 
                        To aid with your answering, you will be provided context delimited by ###. 
                        Your role is to use the context provided to you and respond to the user's question in the most accurate fashion.
                        Every response should be precise, data-driven, and optimized for decision-making. 
                        Structure responses in a way that maximizes usability for quantitative analysis (tables, formulas, charts). 
                        Always cite specific sections from the provided context when possible.
                        If you think that the data in the context is NOT RELEVANT, respond to the user that you do not have the requisite data to respond.
                        DO NOT MAKE UP INFORMATION."""
    

    #Gemini selected because the free-tier allows for easy prototyping
    GEMINI_API_KEY= st.text_input(label="Enter your Gemini API key. You can generate this at https://aistudio.google.com/apikey free of charge")
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        chat = client.chats.create(model="gemini-2.0-flash", 
                                config=types.GenerateContentConfig(system_instruction=system_instruction, 
                                                                    max_output_tokens=500,
                                                                    temperature=0.1))

    
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("How can I help you with your analysis today?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})


            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            #Retrieve the most semantically similar documents using the vectorstore retriever
            most_similar_documents= vectorstore_retriever_object.retrieve(prompt)

            #Gather the most relevant text as context and add it to the user prompt
            sources, concatenated_text= extract_sources_and_text(most_similar_documents=most_similar_documents)
            user_chat_message= f"{prompt}\n Context: ###{concatenated_text}###"
            
            #Generate response using the Google Gemini chat module
            response = chat.send_message(user_chat_message)

            #Format the LLM response to also include the sources from where the data is extracted
            complete_response= f"{response.text}\n\nSources:\n\n{sources}"


            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(complete_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": complete_response})