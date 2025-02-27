from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

# Step 1: Loading Data 

def load_csv_file(data):
    loader = CSVLoader(file_path=data)  # Use file_path instead of path
    rowsxcol = loader.load()
    return rowsxcol

DATA_PATH = 'cars_dataset.csv'  # Path to your CSV file
rowsxcol = load_csv_file(data=DATA_PATH)
#print(rowsxcol)

# Step 2: Creating Chunks

def create_chunks(extracted_data):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data) 
    return text_chunks
text_chunks = create_chunks(extracted_data= rowsxcol)
print("Lenght of text chunks: ", len(text_chunks))

# Step 3: Create Vector embeddings

def get_embedding_model():
    
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DP_FAISS_PATH = "vectorstore/dp_faiss"
dp = FAISS.from_documents(text_chunks, embedding_model)
dp.save_local(DP_FAISS_PATH)        

