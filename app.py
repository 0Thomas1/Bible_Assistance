#from langchain_community import Ollama
#from langchain_community.llms import Ollama
import bs4
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


dataPath = "Bible_Genesis.csv"

#load the data
data_loader = CSVLoader(
    file_path=dataPath,
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Book_id", "Chapter", "Verse", "Text"],
    },
    )
data = data_loader.load()
print("data loaded")



#2. Vectorize the data
embeddings = OllamaEmbeddings(model="llama3")
vectorStore = Chroma.from_documents(documents=data, embedding=embeddings)
print("data vectorized")
  

#3. Create the agent
def ollama_llm(question,context):
    formated_text = f"Question: {question}\n\nContext:{context}"
    response = Ollama.chat(model = "llama3", message=[{'role': 'user', 'content': formated_text}])
    return response['message']['content']


#4. rag
retriver = vectorStore.as_retriever()
print("retriver created")
def combineDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrivededDocs = retriver.retrieve(question)
    context = combineDocs(retrivededDocs)
    return ollama_llm(question,context)

#5. Create the task
print("Loading the model....")
result = rag_chain("Who is the Jesus Christ?")
print(result)