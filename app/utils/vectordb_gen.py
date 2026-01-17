from app.core.config import settings
import pandas as pd
from langchain_community.vectorstores import Chroma
from app.utils import embedding
from langchain_core.documents import Document

class VectorDBGenerator:
    def __init__(self):
        self.vectorDBPath = settings.vectorDBPath
        self.product_data_path = settings.product_data_path
        self.chunks = []
        self.embedding = embedding

    def chunk_preparation(self):
        df = pd.read_csv(self.product_data_path, usecols=['product_name', 'category','price' ,'description', 'specifications', 'order_count'])
        for row in df.iterrows():
            self.chunks.append(f"""Product Name: {row[1]['product_name']}\n Category: {row[1]['category']}\n Price: {row[1]['price']}\n Description: {row[1]['description']}\n Specifications: {row[1]['specifications']}\n Order Count: {row[1]['order_count']}""")
        self.chunks.append(f"""Following are the available products in system:\n {list(set(df['product_name']))}""")
        return self.chunks
    
    def save_to_chroma(self, chunks):
        documents = [Document(page_content=chunk) for chunk in chunks]
        db = Chroma.from_documents(
            documents,
            self.embedding,
            persist_directory=self.vectorDBPath
        )
        db.persist()
    
    def generate_vector_db(self):
        chunks = self.chunk_preparation()
        self.save_to_chroma(chunks)
        return "Vector DB generated and saved successfully."
