from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from typing import Callable

load_dotenv()
text_embedding_type= os.environ.get("TEXT_EMBEDDING_TYPE")
text_embeddings_model= os.environ.get("TEXT_EMBEDDING_MODEL")
PERSIST_DIR= os.environ.get("PERSIST_DIRECTORY")
n_ctx= os.environ.get("N_CTX")
chain_type= os.environ.get("CHAIN_TYPE")
prompt_file= os.environ.get("PROMPT_FILE")
def fetch_embedding_model()-> tuple[HuggingFaceEmbeddings | LlamaCppEmbeddings | Callable]:
    match text_embedding_type:
        case "HF":
            model= HuggingFaceEmbeddings(model_name= text_embeddings_model)
            return model, model.client.encode
        case "LlamaCpp":
            model= LlamaCppEmbeddings(model_path= text_embeddings_model, n_ctx= n_ctx) # n_gpu_layers could be added
            return model, lambda input_ : model.client.embed(input_) if isinstance(input_, str) else [model.client.embed(e) for e in input_]

        case _: 
            raise ValueError(f"Unknown embedding type given: {text_embedding_type}")

def get_promptTemplate_kwargs() -> dict[str, PromptTemplate]:
    ''' It takes the prompt template as input & returns the prompt a/c to chain_type '''
    try:
        with open(prompt_file, 'r') as file:
            prompt= file.read()
        file.close()
    except FileNotFoundError:
        print(f"File Path: {prompt_file} could not be found!")
    ''' Chain Type is taken as "stuff". In future release "refine" might be added. '''
    return {'prompt': PromptTemplate(template= prompt, input_variables= ['chat_history', 'question'], validate_template=True)}

