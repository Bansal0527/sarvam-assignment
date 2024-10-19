from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 static
templates = Jinja2Templates(directory="static")

# Initialize components
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
loader = PyPDFLoader("Data/ncert.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# Function to classify if a query needs the VectorDB or other tools
def call_llm_api(query):
    """
    This function uses the already initialized `llm` to decide whether to query the VectorDB.
    """
    try:
        llm_prompt = f"Does the following query require retrieving information from the database?\nQuery: {query}\nAnswer with Yes or No."
        response = llm.invoke(llm_prompt)

        # Access the 'content' attribute of the response to get the actual text
        llm_response_text = response.content  # Extracting the text content
        logging.info(f"LLM Response for query classification: {llm_response_text}")

        if llm_response_text and "yes" in llm_response_text.lower():
            return True
        return False
    except Exception as e:
        logging.error(f"Error in call_llm_api: {str(e)}")
        raise HTTPException(status_code=500, detail="Error while querying the LLM")


# Summarization tool
def some_summarizer_api(text):
    """
    Simple summarization function using the same LLM model.
    """
    try:
        summarize_prompt = f"Summarize the following text:\n{text}"
        summary = llm.invoke(summarize_prompt)
        logging.info(f"LLM Response for summarization: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Error in summarizer API: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during summarization")


# Calculator tool
def perform_calculation(query):
    """
    Basic calculator function to evaluate math expressions.
    """
    try:
        result = eval(query)  # Be careful with eval() in production
        return f"The result is {result}"
    except Exception as e:
        logging.error(f"Error in perform_calculation: {str(e)}")
        return "I couldn't perform the calculation."


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/query")
async def query(text: str):
    try:
        logging.info(f"Received query: {text}")

        # First, let the LLM decide if the query needs a VectorDB lookup
        if call_llm_api(text):
            response = rag_chain.invoke(text)
        elif "summarize" in text.lower():
            logging.info("Invoking summarizer.")
            response = some_summarizer_api(text)
        # elif any(char.isdigit() for char in text):
        #     logging.info("Performing calculation.")
        #     response = perform_calculation(text)
        else:
            response = "I'm not sure how to help with that query."

        return JSONResponse(content={"text": response})

    except Exception as e:
        logging.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

