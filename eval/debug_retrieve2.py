import sys
sys.path.insert(0, "src")

from langchain_huggingface import HuggingFaceEmbeddings
from retriever import Retriever

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

# 不用rerank，直接看semantic collection的原始向量召回，扩大到10条
r = Retriever(collection_name="langchain_semantic", embedding_model=embedding_model, use_rerank=False)
query = "使用 LangChain 的 @tool 装饰器创建工具时，docstring 有什么作用？"
result = r.retrieve(query, top_k=10)

print(f"原始向量召回 {len(result.documents)} 个文档（未rerank）")
for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
    print(f"\n[{i+1}] score={score:.4f} source={doc.metadata.get('source')} 长度={len(doc.page_content)}字符")
    print(doc.page_content[:100])