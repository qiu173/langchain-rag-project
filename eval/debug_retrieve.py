import sys
sys.path.insert(0, "src")

from langchain_huggingface import HuggingFaceEmbeddings
from retriever import Retriever

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

r = Retriever(collection_name="langchain_semantic", embedding_model=embedding_model, use_rerank=True)

query = "使用 LangChain 的 @tool 装饰器创建工具时，docstring 有什么作用？"
result = r.retrieve(query, top_k=5)

print(f"检索到 {len(result.documents)} 个文档")
for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
    print(f"\n[{i+1}] score={score:.4f} source={doc.metadata.get('source')}")
    print(doc.page_content[:150])