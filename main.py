import streamlit as st
import pandas as pd
from openai import OpenAI
import tiktoken
from scipy import spatial
from PyPDF2 import PdfReader
import re
import os
import zipfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

client = OpenAI(api_key=st.secrets['api_key'])

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n"," ")
    return client.embeddings.create(input=[text],model=model).data[0].embedding

def remove_stopwords(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def split_paragraph(paragraph):
    # Split the paragraph into lines using regular expressions
    lines = re.split(r"[\n\r]+", paragraph)

    # Remove lines with less than 20 characters
    lines = [line for line in lines]

    # Group consecutive lines into paragraphs
    paragraphs = []
    current_paragraph = []

    for line in lines:
        if line:
            current_paragraph.append(line)
        else:
            # Check if current paragraph has any content
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []

    # Add the last paragraph if it has any content
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs

def num_tokens_from_string(string: str, encoding_name: str = "text-embedding-3-small") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_long_text(text):
    MAX_TOKENS = 8191
    vectors_parent = []
    num_tokens = num_tokens_from_string(text)
    num_chunks = (num_tokens + MAX_TOKENS - 1) // MAX_TOKENS  # Ceiling division

    for i in range(num_chunks):
        start_idx = i * MAX_TOKENS
        end_idx = min((i + 1) * MAX_TOKENS, num_tokens)
        chunk_text = text[start_idx:end_idx]
        vectors = get_embedding(chunk_text) # Vector of length 1536
        vectors_parent.append({"text": chunk_text, "vector": vectors})
        
    return vectors_parent

def strings_ranked_by_relatedness(query: list, df: pd.DataFrame, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n: int = 10) -> tuple[list[str], list[float]]:
    strings_and_relatednesses = [(row["text"], relatedness_fn(query, row["vector"])) for _, row in df.iterrows()]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return split_paragraph(text)

def main():
    st.title("HUBX Data Room Analyzer")
    st.write("Upload Single File to Analyze them.")
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    df_list = []
    
    # Wrap input handling inside st.form context manager
    with st.form(key='my_form'):
        if uploaded_files:
            for file in uploaded_files:
                st.write(f"## Analyzing {file.name}")
                file_contents =extract_text_from_pdf(file)
                
                for file_content in file_contents: 
                    print(file_content)           
                    file_content = remove_stopwords(str(file_content))
                    num_tokens = num_tokens_from_string(file_content)
                    if num_tokens > 1:
                        if num_tokens < 8191:
                            vectors = get_embedding(file_content)
                            df = pd.DataFrame({"text": [file_content], "vector": [vectors]})
                            df_list.append(df)
                        else:
                            vectors = process_long_text(file_content)
                            for vector in vectors:
                                df = pd.DataFrame({"text": [vector["text"]], "vector": [vector["vector"]]})
                                df_list.append(df)     
                    else:
                        print("TOKEN IS ",num_tokens,"Content was ",file_content)
                        print("Skipping")

            user_input = st.text_input("What's your Query?")
            submitted = st.form_submit_button(label='Submit')

            # Process the user input when the form is submitted
            if submitted:
                if user_input:
                    df = pd.concat(df_list)  
                    st.write("You entered: ",user_input)
                    user_input_embedding  = get_embedding(user_input)
                    strings = strings_ranked_by_relatedness(user_input_embedding,df)
                    text_content = str(strings)
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        temperature=1,
                        presence_penalty=-1,
                        messages=[{"role": "system", "content": f""" You are an RAG assistant. If you do not have a clear answer then say "You Dont Know". Give concise and simple response to user queries Keeping in mind the context Given below. Here are the top related documents to the user queries:  """+text_content},{"role":"user","content":user_input}],)
                    index = completion.choices[0].message.content
                    st.write(index)

        
        # st.title("Folder Upload Example")

    uploaded_file = st.file_uploader("Upload zip file", type="zip")

    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract the contents of the zip file
        with zipfile.ZipFile(uploaded_file.name, "r") as zip_ref:
            zip_ref.extractall("uploaded_folder")
        
        st.success("Folder uploaded successfully!")
        
        # List the contents of the uploaded folder
        st.write("Contents of the uploaded folder:")
        for root, dirs, files in os.walk("uploaded_folder"):
            for file in files:
                st.write(os.path.join(root, file))
                with open(os.path.join(root, file), 'rb') as f:
                    content = str(extract_text_from_pdf(f))
                    path = os.path.join(root,file)
                    text_content = f"Path to file: {path} \n Contents of File: {content}"
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        temperature=1,
                        presence_penalty=-1,
                        messages=[{"role": "system", "content": """You will get the path and the contents of each file, You have to summarize these files keeping in mind the path at which it's saved for context. Reply in markdown with bullet points to make it easy to read. Summarize the file in about two paragraphs. List and Mention important details such as dates, entities and a general idea about each document. Only Reply with a summary of the documents. """},{"role":"user","content":text_content}],)
                    index = completion.choices[0].message.content
                    st.write(index)

                


if __name__ == "__main__":
    main()
