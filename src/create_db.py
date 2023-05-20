from typing import Callable, Any
import contextlib
import multiprocessing
import os
import glob
import sys
from hashlib import md5
from pathlib import Path
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models 
from load_env import fetch_embedding_model
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import ProgressBar
import shutil

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("spawn", force=True)
load_dotenv()
EXT= {
        ".txt" : (TextLoader, {"encoding":"utf8"}),
        ".csv" : (CSVLoader, {}),
        ".pdf" : (PDFMinerLoader, {})
    }
class CreateDB:
    def __init__(self,**kwargs):
        self.persist_directory= os.environ.get('PERSIST_DIRECTORY')
        self.source_directory= os.environ.get('SOURCE_DIRECTORY')
        self.n_batch= int(os.environ.get('N_BATCH'))
        self.collection_name= os.environ.get('COLLECTION_NAME')
        self.n_thread= int(os.environ.get('N_THREAD'))
        self.check_storage= []
        self.verbose= kwargs['verbose']
        self.encode_fun= None

    def load_one_document(self, filepath: Path)-> Document:
        ext= "." + filepath.rsplit('.',1)[-1] # -1 significant
        if ext in EXT:
            loader_class, loader_args= EXT[ext]
            loader= loader_class(filepath, **loader_args)
            return loader.load()
        raise ValueError(f"UNSUPPORTED FILE EXTENSION: {ext}")

    def embed_documents(self, embedding_function: Callable, documents: list[Document]) -> list[tuple[Any, Document]]:
        if self.verbose:
            print(f"Processing {len(documents)} chunks")
        embeddings= embedding_function([doc.page_content for doc in documents]).tolist()
        return list(zip(embeddings, documents))

    ''' Creating the vector using Qdrant'''
    def store_embeddings(self,embedding_and_document : list[tuple[Any, Document]],force: bool = False) -> None:
        self.check_storage+= embedding_and_document
        if not force and len(self.check_storage)<self.n_batch:
            return
        client= QdrantClient(path= self.persist_directory, prefer_grpc=True)
        try:
            client.get_collection(self.collection_name)
        except ValueError:
            vector_size= max(len(e[0]) for e in self.check_storage)
            print(f"Creating new Collection of vector size= {vector_size}.")
            client.recreate_collection(collection_name=self.collection_name, 
                                       vectors_config=models.VectorParams(
                size= vector_size,distance=models.Distance['COSINE'])) 
            # Note: The vector distance is defined by cos function
            # Will be utilized when calculating similarity_search
        print(f"Saving {len(self.check_storage)} chunks.")
        embeddings, texts, metadatas = (
            [e[0] for e in self.check_storage],
            [e[1].page_content for e in self.check_storage],
            [e[1].metadata for e in self.check_storage],
        )
        client.upsert(
            collection_name=self.collection_name,
            points=models.Batch.construct(
                ids=[md5(text.encode("utf-8")).hexdigest() for text in texts],
                vectors=embeddings,
                payloads=[{"page_content": text, "metadata": metadatas[i]} for i, text in enumerate(texts)],
            ),
        )
        collection = client.get_collection(self.collection_name)
        self.awaiting_storage = []
        if self.verbose:
            print(f"Saved, the collection now holds {collection.points_count} documents.")

    def process_one_doc(self,filepath: Path) -> list[tuple[Any, Document]]:
        doc= self.load_one_document(filepath)
        if not doc:
            return None 
        text_split= RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 30, separators= ['\n\n','\n', ' ',''])
        data= text_split.split_documents(doc)
        res= self.embed_documents(self.encode_fun, data)
        return res
    
    def create(self) -> None:
        _, self.encode_fun= fetch_embedding_model()
        all_files= []
        for ext in EXT:
            all_files.extend(glob.glob(os.path.join(self.source_directory, f'**/*{ext}'), recursive= True))
        with ProgressBar() as pb:
            with multiprocessing.Pool(self.n_thread) as pool:
                for embeddings in pb(pool.imap_unordered(self.process_one_doc, all_files), total=len(all_files)):
                    if embeddings is None:
                        continue
                    self.store_embeddings(embeddings)
            self.store_embeddings(embeddings, force=True)
        print("Done")
            
def main(clean_db: str):
    db= CreateDB(verbose= True)
    session= PromptSession()
    
    if os.path.exists(db.persist_directory):
        if clean_db.lower() == "y" or (clean_db == "n" and session.prompt("\nDelete current database?(Y/N)").lower() == "y"):
            print("Deleting db...")
            shutil.rmtree(db.persist_directory)
        elif clean_db.lower() == "n":
            print("Adding to db...")
    db.create()
if __name__=="__main__":
    clean_db= sys.argv[1] if len(sys.argv)> 1 else "n"
    main(clean_db)
