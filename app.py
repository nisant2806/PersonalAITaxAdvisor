import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.chat_models import init_chat_model 
from langchain.agents import create_agent

load_dotenv("app.env")

pdf_files = [
    "D:\\gen ai class\\sem 3\\Building a Personal Finance Advisor\\Personal Tax AI Advisor\\CBDT_e-Filing_ITR 4_Validation Rules_AY 2026-27.pdf",
    "D:\\gen ai class\\sem 3\\Building a Personal Finance Advisor\\Personal Tax AI Advisor\\Common ITR Filing FAQs AY 2024-25.pdf",
    "D:\\gen ai class\\sem 3\\Building a Personal Finance Advisor\\Personal Tax AI Advisor\\ITR-7 FAQ.pdf",
    "D:\\gen ai class\\sem 3\\Building a Personal Finance Advisor\\Personal Tax AI Advisor\\New vs. Old Regime FAQs.pdf",
    "D:\\gen ai class\\sem 3\\Building a Personal Finance Advisor\\Personal Tax AI Advisor\\Securities Market Booklet (1).pdf",
]

add_documents = []
for doc in pdf_files:
    loader = PyPDFLoader(doc)
    data = loader.load()
    add_documents.extend(data)

text_split = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
chunks = text_split.split_documents(add_documents)

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="tax_docs",
    persist_directory="./tmp/chroma_tax_db"
)

@tool
def search_tax_docs(query: str) -> str:
    """Search official income tax documents and SEBI financial education guides for Section 80C/80D deductions, old vs new tax regime comparison, ITR filing guidance, and tax saving investments."""
    retrieve_doc = vector_store.similarity_search(query,k=3)

    if not retrieve_doc:
        return "No relevant information found in the knowledge base."

    context = ""
    for doc in retrieve_doc:
        context+= f"Page Content: {doc.page_content}\n\n"
    return context

api_key = os.getenv("GEMINI_API_KEY")
model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    api_key = api_key
)

system_prompt = """You are a Personal Tax Advisor for Indian citizens.

You have access to these tools:
- search_tax_docs: Search official Indian government documents for tax saving
  options (80C, 80D), old vs new tax regime, ITR filing, and investment guidance

Help the user by looking up the relevant data using your tools and giving clear,
specific answers with actual numbers and rates. Always mention the source of
your information. All monetary values should be in Indian Rupees (₹) unless
specified otherwise."""

agent = create_agent(
    model = model,
    tools = [search_tax_docs],
    system_prompt = system_prompt
)
sample_query_80c = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What are the deductions available under Section 80C?",
            }
        ]
    }
)
print("80C Sample Output:", sample_query_80c)

sample_query_regime = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the difference between old and new tax regime?",
            }
        ]
    }
)
print("Regime Sample Output:", sample_query_regime)

def tax_advisor(question):
    response = agent.invoke({
        "messages":[
            {
                "role":"user",
                "content":question
            }
        ]
    })
    return response["messages"][-1].content[0]['text']

demo = gr.Interface(
    fn=tax_advisor,
    inputs=gr.Textbox(lines=2, placeholder="Ask a tax question...", label="Question"),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="Personal Tax AI Advisor",
    description="Ask about income tax saving, 80C/80D deductions, old vs new regime, and ITR filing.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=True)