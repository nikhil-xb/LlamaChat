from preprocess import preProcess
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS
import glob
import os
import tiktoken
load_dotenv()
EXT= {
        ".txt" : (TextLoader, {"encoding":"utf8"}),
        ".csv" : (CSVLoader, {}),
        ".pdf" : (PDFMinerLoader, {})
    }
tokenizer= tiktoken.get_encoding('cl100k_base')
def tiktoken_len(text):
    tokens= tokenizer.encode(text, disallowed_special=())
    return len(tokens)
def load_document(filepath):
    ext= "." + filepath.rsplit('.',1)[-1] # -1 significant
    if ext in EXT:
        loader_class, loader_args= EXT[ext]
        loader= loader_class(filepath, **loader_args)
        return loader.load()[0]
    raise ValueError(f"UNSUPPORTED FILE EXTENSION: {ext}")
    
def load_all(source_dir):
    all_files= []
    for ext in EXT:
        all_files.extend(glob.glob(os.path.join(source_dir, f'**/*{ext}'), recursive= True))
    return [load_document(filepath) for filepath in all_files]

def main():
    persist_directory= os.environ.get('PERSIST_DIRECTORY')
    source_directory= os.environ.get('SOURCE_DIRECTORY')
    embeddings_model_name= os.environ.get('EMBEDDINGS_MODEL_NAME')
    context_num= os.environ.get('n_ctx')

    print("Loading Training Data...")
    documents= load_all(source_directory)
    text_split= RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 30, separators= ['\n\n','\n', ' ',''])
    data= text_split.split_documents(documents)
        
    #print(data)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(data)} chunks of text max= 500 characters of each")
    # Creating Embeddings
#    llama= LlamaCppEmbeddings(model_path= llama_embeddings, n_ctx=context_num)
    embeddings= HuggingFaceEmbeddings(model_name= embeddings_model_name)   
    #Creating Vector Database from Chroma
    vectorstore= Chroma(embedding_function=embeddings, 
            client_settings= CHROMA_SETTINGS, persist_directory= persist_directory)
    vectorstore.add_documents(documents= data, embedding= embeddings)
    vectorstore.persist()
    vectorstore= None

if __name__=="__main__":
    main()
