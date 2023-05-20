from langchain.llms import LlamaCpp, GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from load_env import get_promptTemplate_kwargs, fetch_embedding_model
import qdrant_client
from langchain.vectorstores import Qdrant
from prompt_toolkit import PromptSession
from dotenv import load_dotenv
import os
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")


class YouGPT:
    def __init__(self, **kwargs):
        self.persist_directory = os.environ.get('PERSIST_DIRECTORY')
        self.model_type = os.environ.get('MODEL_TYPE')
        self.model_path = os.environ.get('MODEL_PATH')
        self.model_n_ctx = os.environ.get('n_ctx')
        self.collection_name= os.environ.get('COLLECTION_NAME')
        self.embeddings= kwargs['embeddings']

        callbacks = [StreamingStdOutCallbackHandler()]

        self.qdrant_client= qdrant_client.QdrantClient(path=self.persist_directory, prefer_grpc=True)
        self.qdrant_langchain= Qdrant(client=self.qdrant_client,
                                      collection_name=self.collection_name,
                                      embeddings=self.embeddings)
        
        match self.model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=self.model_path, n_ctx=self.model_n_ctx, callbacks=callbacks, verbose=False)
            case "GPT4All":
                llm = GPT4All(model=self.model_path, n_ctx=self.model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            case _:
                raise ValueError(f"Model {self.model_type} not supported!")
            
        self.llm= llm
        retriever= self.qdrant_langchain.as_retriever(search_type="mmr")
        memory= ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        val= get_promptTemplate_kwargs()
        self.llm_chain= ConversationalRetrievalChain.from_llm(llm= llm, retriever= retriever, 
                                                         condense_question_prompt= val['prompt'],
                                                         chain_type="stuff", 
                                                         memory=memory,
                                                         get_chat_history= self.get_chat_history
                                                         )
    
    def get_chat_history(self,inputs) -> str:
        res = []
        for human, ai in inputs:
            res.append(f"User->{human}\nYouGPT->{ai}")
        return "\n".join(res)
    
    def onMessage(self,query, history):
        "Takes a single prompt and compute the answer"
        result= self.llm_chain({'question':query, 'chat_history': history})
        # context.append([doc.page_content for doc in result['source_documents']])
        return result

def main():
    
    kwargs= {'embeddings': fetch_embedding_model()[0]}
    gpt= YouGPT(**kwargs)
    while True:
        query= input("User>")
        if query=="exit":
            print("Exiting....")
            break;
        elif not query:
            print("Empty Query...")
            continue;
        history= []
        result= gpt.onMessage(query,history)
        print(f"YouGPT>{result['answer']}")
if __name__=="__main__":
    main()
