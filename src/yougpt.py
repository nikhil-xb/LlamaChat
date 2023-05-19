from langchain.llms import LlamaCpp, GPT4All
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
import chromadb
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS
import os
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('n_ctx')


embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
# client= chromadb.Client(settings=CHROMA_SETTINGS)
# collection= client.get_collection(collection_name= 'langchain_store', embedding_function=embeddings)
vectorstores= Chroma(persist_directory=persist_directory, embedding_function= embeddings)
retriever= vectorstores.as_retriever()
callbacks = [StreamingStdOutCallbackHandler()]
match model_type:
    case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    case default:
        print(f"Model {model_type} not supported!")
        exit;

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"User->{human}\nYouGPT->{ai}")
    return "\n".join(res)

with open('prompts.txt') as file:
    PromptTemp= file.read()
file.close()

prompt= PromptTemplate(template= PromptTemp, input_variables= ['chat_history', 'question'], validate_template=True)
memory= ConversationBufferMemory(memory_key='chat_history', return_messages=True)

llm_chain= ConversationalRetrievalChain.from_llm(llm= llm, retriever= retriever, condense_question_prompt= prompt,                                                  chain_type="stuff", memory=memory,get_chat_history= get_chat_history)
while True:
    query= input("\n>User: ")
    if query=="exit":
          break
    context= []
    history= []
    result= llm_chain({'question':query, 'chat_history': history})
    context.append([doc.page_content for doc in result['source_documents']])
    print(f"\n> YouGPT: {result['answer']}")




