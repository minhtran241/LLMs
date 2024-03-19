from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_params

client = MongoClient(key_params.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_key=key_params.OPENAI_API_KEY)

vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)


def query_data(query):
    docs = vectorStore.similarity_search(
        query, K=1
    )  # Convert input query into a vector and find the most similar document. K=1 means return the most similar document. K is the number of documents to return.
    as_output = docs[0].page_content  # Get the content of the most similar document.

    llm = OpenAI(
        openai_api_key=key_params.OPENAI_API_KEY, temperature=0
    )  # Create a language model, temperature=0 means the model will be deterministic. Temperature is a parameter that controls the randomness of the model's output.
    retriever = (
        vectorStore.as_retriever()
    )  # Convert the vector store into a retriever. Retriever is a class that can be used to retrieve documents from the vector store.
    qa = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever
    )  # This chain is then executed with original query. LLM generating response based on the query and retriever finding the most relevant document. "stuff" document chain type means the chain will take all relevant documents, insert them all into a prompt, and pass that prompt to the LLM.
    retriever_output = qa.run(
        query
    )  # Execute the chain with the query. This will return the response from the LLM and the most relevant document.
    return as_output, retriever_output


with gr.Block(
    theme=Base(), title="Question Answering using Vector Search + RAG"
) as demo:
    gr.Markdown(
        """
        # Question Answering using Vector Search + RAG
        """
    )
    textbox = gr.Textbox(lines=2, label="Ask a question:")
    with gr.Row():
        button = gr.Button(text="Ask", variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Most similar document:")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Answer:")
    button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()
