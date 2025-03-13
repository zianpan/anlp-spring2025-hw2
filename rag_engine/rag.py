# %%
# RAG_System.ipynb

# ============================
# 1. Install Required Packages
# ============================
# You might already have some or all of these. If so, you can skip or comment them out.
# %pip install langchain transformers chromadb sentence-transformers accelerate bitsandbytes  # etc.

import os
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import shutil
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import gc
from tqdm import tqdm, trange
import argparse

parser = argparse.ArgumentParser(description="Compare multiple RAG system outputs against multiple reference answers")
parser.add_argument("--llm", type=str, required=True, choices=["falcon", "llama3", "deepseek-r1", "phi-4","qwen2"], help="Local LLM model ID")
parser.add_argument("--qa_file", type=str, choices=["qa400", "qa2500"], required=True, help="Name of QA file")
parser.add_argument("--chunk_size", type=int, default=512, required=True, help="Size of chunks for processing input")
parser.add_argument("--chunk_overlap", type=int, default=100, required=True, help="Size of chunks for processing input")
parser.add_argument("--retriever_top_k", type=int, default=4, required=True, help="How many top documents to retrieve")
parser.add_argument("--reload_vectors_db", type=int, choices=[0,1], required=True, help="Reload vectors database")
# parser.add_argument("--reload_vectors_db", action="store_true", help="Reload vectors database")

args = parser.parse_args()

# gc.collect()
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TEXT_DATA_PATH = ["../data/zianp", "../data/dunhanj"] 
ROW_EVENT_PATH = ['../data/nicolaw']
STATIC_WEB_CSV_PATH = '../data/texts_urls_filtered.csv'
custom_cache_dir = "/mnt/new_volume"

# Choose an embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_MAP = {"falcon": "tiiuae/falcon-7b-instruct"
                ,"llama3": "meta-llama/Llama-3.1-8B-Instruct"
                ,"deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                ,"phi-4": "unsloth/phi-4-bnb-4bit"
                ,"qwen2": "Qwen/Qwen2-7B-Instruct"}

LLM_MODEL_ID = LLM_MODEL_MAP[args.llm]
LLM_NAME = LLM_MODEL_ID.split("/")[-1]
data_file = args.qa_file
test_data_path ="../annotations/{}.csv".format(data_file)

retriever_top_k = args.retriever_top_k
CHUNK_SIZE = args.chunk_size  
CHUNK_OVERLAP = args.chunk_overlap
RELOAD_VECTORS_DB = True if args.reload_vectors_db == 1 else False


gc.collect()
torch.cuda.empty_cache()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "mps" 

# # %%
# # ============================
# # 2. Configuration
# # ============================
# # Path to data folder
# TEXT_DATA_PATH = ["../data/zianp", "../data/dunhanj"] 
# ROW_EVENT_PATH = ['../data/nicolaw']
# STATIC_WEB_CSV_PATH = '../data/texts_urls_filtered.csv'
# custom_cache_dir = "/mnt/new_volume"

# # Choose an embedding model.
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # Choose a local LLM model.
# # LLM_MODEL_ID = "tiiuae/falcon-7b-instruct"
# # LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# # LLM_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# LLM_MODEL_ID = "unsloth/phi-4-bnb-4bit"
# LLM_NAME = LLM_MODEL_ID.split("/")[-1]
# data_file = "qa400"
# test_data_path ="../annotations/{}.csv".format(data_file)

# retriever_top_k = 4
# CHUNK_SIZE = 512  
# CHUNK_OVERLAP = 100
# RELOAD_VECTORS_DB = False



# %%
# Classify files in the folder

files_txt_path = []
files_csv_path = []
files_event_path = []

for DATA_PATH in TEXT_DATA_PATH:
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.txt'):
                files_txt_path.append(os.path.join(root, file))
            elif file.endswith('.csv'):
                files_csv_path.append(os.path.join(root, file))

for DATA_PATH in ROW_EVENT_PATH:
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.txt'):
                files_event_path.append(os.path.join(root, file))




# %%

# ============================
# 2. Load Files with Different Strategies
# ============================
all_documents = []

# Load Dunhan CSV
test_df = pd.read_csv(STATIC_WEB_CSV_PATH)
for index, row in test_df.iterrows():
    
    all_documents.append(Document(page_content=row['TEXT'], metadata={"source": row['URL']}))

# Load all files in the directory
for file_path in files_txt_path:
    loader = TextLoader(file_path, encoding="utf-8")
    doc = loader.load()  # Load entire file as one document
    all_documents.append(Document(page_content=doc[0].page_content, metadata={"source": file_path}))

for file_path in files_csv_path:
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    for index, row in df.iterrows():
        row_text = f"{filename} | " + " | ".join(f"{col}: {row[col]}" for col in df.columns)
        metadata = {"source": filename, "row_id": index}
        all_documents.append(Document(page_content=row_text, metadata=metadata))


# OPTIOANL function for processing files row by row
    # ✅ Load row by row (structured data)
for file_path in files_event_path:
    with open(file_path, "r", encoding="utf-8") as file:
        for row_id, line in enumerate(file):
            line = line.strip()
            if line:  # Ignore empty lines
                all_documents.append(Document(page_content=line, metadata={"source": filename, "row_id": row_id}))


print(f"Loaded {len(all_documents)} raw documents from {len(os.listdir(DATA_PATH))} files.")

# ============================
# 3. Split Longer Documents for Better Retrieval
# ============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

split_documents = []
for doc in all_documents:
    chunks = text_splitter.split_text(doc.page_content)  # Split if needed
    for chunk in chunks:
        split_documents.append(Document(page_content=chunk, metadata=doc.metadata))

print(f"Total {len(split_documents)} final chunks prepared for vector storage.")


# %%

# ============================
# 4. Create Embeddings
# ============================
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, cache_folder = custom_cache_dir)
print("Embeddings loaded successfully.")

# ============================
# 5. Manage Vector Store
# ============================
persist_directory = "chroma_db"

# Check if the vector store exists and delete it if necessary
if RELOAD_VECTORS_DB:

    if os.path.exists(persist_directory):
        print("Vector store exists. Deleting existing database...")
        shutil.rmtree(persist_directory)  # Deletes the existing database folder

    # Recreate the vector store
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
else:
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Local Vector store loaded successfully.")

vectorstore.persist()
print("Vector store recreated and persisted.")



# %%

# ============================
# 6. Set Up the LLM (Falcon 7B Instruct)
# ============================
# Load the tokenizer and model
print(f"Loading {LLM_MODEL_ID}; this may take some time...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True, cache_dir=custom_cache_dir)
tokenizer.pad_token = tokenizer.eos_token  
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map= device,           # automatically place model layers on available GPU
    trust_remote_code=True,
    cache_dir=custom_cache_dir
)


# %%
# Create a text-generation pipeline
pipeline_llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=20,
    temperature= 0.1,       # Lower temperature for more factual answers
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
)

# Wrap the pipeline in a LangChain LLM
llm = HuggingFacePipeline(pipeline=pipeline_llm)


# Customized Prompt

QA_Prompt = """
You are an expert assistant answering factual questions about Pittsburgh or Carnegie Mellon University (CMU). 
Use the retrieved context to give a detailed and helpful answer. If the provided context does not contain the answer, leverage your pretraining knowledge to provide the correct answer. 

Important Instructions:
- Answer concisely without repeating the question.
- Use the provided context if relevant; otherwise, rely on your pretraining knowledge.
- Do **not** use complete sentences. Provide only the word, name, date, or phrase that directly answers the question. For example, given the question "When was Carnegie Mellon University founded?", you should only answer "1900".

Retrieved Context:
---
{context}
---

Examples:

Question: In less than 5 words, Who is Pittsburgh named after? 
Answer: William Pitt \n
Question: In less than 5 words, What famous machine learning venue had its first conference in Pittsburgh in 1980? 
Answer: ICML \n
Question: In less than 5 words, What musical artist is performing at PPG Arena on October 13? 
Answer: Billie Eilish \n

Now it's your turn. Please answer the following question based on the above context. Remember to answer as short as possible. 

Question: In less than 5 words, {question}
Answer:
"""

custom_prompt = PromptTemplate(template=QA_Prompt, input_variables=["context", "question"])


# ============================
# 7. Create the RetrievalQA Chain
# ============================
retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_top_k})


def ask_question(query: str):
    """
    Run a query through the RAG pipeline and return the generated answer along with the source documents.
    
    Args:
        query (str): The user’s question.

    Returns:
        answer (str): The generated answer.
        sources (list): List of retrieved documents used to generate the answer.
    """
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)
    # print(f"Retrieved {len(retrieved_docs)} documents.")
    
    # Extract text from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # print(f"Context length: {len(context)} characters.")
    # print('------ START CONTEXT ------')
    # print(context)
    # print('------ END CONTEXT ------')

    # Format the input using the QA_Prompt
    formatted_prompt = QA_Prompt.format(context=context, question=query)
    
    # Generate response using the LLM
    result = llm(formatted_prompt)  # Pass the fully formatted input
    answer = result.replace(formatted_prompt, "").strip()
    # Extract answer and sources
    answer = answer.strip()  # Ensure clean output
    return answer, retrieved_docs  # Return both answer and retrieved documents


# %%
df = pd.read_csv(test_data_path)

# %%

questions = []
references = []
answers = []
sources = []
errors = []
full = df.shape[0]
subset = full

for i in trange(full):
    row = df.iloc[i]

    answer = "I don't know."

    try:
        answer, retrieved_docs = ask_question(row['question'])
    except:
        errors.append((row['question']))
        continue
    # print(answer)
    answer = answer.strip()
    # print(answer)
    answer = answer.split('\n')[0]
    answers.append(answer)
    questions.append(row['question'])
    sources.append(retrieved_docs)
    references.append(row['reference_answer'])


df_ans = pd.DataFrame({'question': questions, 'answer': answers, 'reference_answer': references, 'source': sources})

# %%
df_ans

# %%
df_ans.to_csv(f'../results/test_{data_file}_{LLM_NAME}_ck{CHUNK_SIZE}_ckolap{CHUNK_OVERLAP}_retop{retriever_top_k}.csv', index=False)
