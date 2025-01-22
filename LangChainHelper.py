from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from secret_key import openapi_key

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=openapi_key)

#video_url = "https://www.youtube.com/watch?v=QbuNXQOf9vQ"
def createVectorDBFromYouTubeURL(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = text_splitter.split_documents(transcript)
    print("length of docs",len(docs))
    db = FAISS.from_documents(docs,embeddings)
    return db

def get_resp_from_query(db,query,k=4):
    #k is set to 4 as chunk size is 1000 and model can support 4000 tokens approx. So, we can send 4 chunks of 1000 each
    semanticDocs =db.similarity_search(query,k=k)
    docs_page_content = " ".join([sd.page_content for sd in semanticDocs])
    llm = OpenAI(model="gpt-3.5-turbo-instruct",openai_api_key=openapi_key)
    prompt = PromptTemplate(
        input_variables = ['question','docs'],
        template = """
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.
        Answer the following {question}:
        By searching the following transcript {docs} 
        Only use the factual information from the video transcript. If you do not have enough information, say "I do not know".
        Your answer should be detailed.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n"," ")
    return response
