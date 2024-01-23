import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings('ignore')


embeddings_open = OllamaEmbeddings(model="mistral")
# OpenAI embeddings
#embedding = OpenAIEmbeddings()

llm_open = Ollama(  model="mistral",
                    #model='Llama2',
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

