from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings


model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding =  HuggingFaceEmbeddings(
                                model_name=settings.MODEL_NAME,
                                model_kwargs=model_kwargs,
                                encode_kwargs=encode_kwargs
                            )