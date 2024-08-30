import chromadb
from chromadb.config import Settings
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sentence_transformers import SentenceTransformer

from src.config import EmbeddingModulePath

class MyVectorDB:
    def __init__(self, collection_name):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        model = SentenceTransformer(EmbeddingModulePath)

        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.bge_model = model

    def add_documents(self, filename, page_numbers=None, min_line_length=1, metadata={}):
        paragraphs = self.extract_text_from_pdf(filename, page_numbers, min_line_length)
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(paragraphs),  # 每个文档的向量
            documents=paragraphs,  # 文档的原文
            ids=[f"id{i}" for i in range(len(paragraphs))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results

    def embedding_fn(self, paragraphs):
        '''文本向量化'''
        doc_vecs = [
            self.bge_model.encode(doc, normalize_embeddings=True).tolist()
            for doc in paragraphs
        ]
        return doc_vecs

    def extract_text_from_pdf(self, filename, page_numbers=None, min_line_length=1):
        '''从 PDF 文件中（按指定页码）提取文字'''
        paragraphs = []
        buffer = ''
        full_text = ''
        # 提取全部文本
        for i, page_layout in enumerate(extract_pages(filename)):
            # 如果指定了页码范围，跳过范围外的页
            if page_numbers is not None and i not in page_numbers:
                continue
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text() + '\n'
        # 按空行分隔，将文本重新组织成段落
        lines = full_text.split('\n')
        for text in lines:
            if len(text) >= min_line_length:
                buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
            elif buffer:
                paragraphs.append(buffer)
                buffer = ''
        if buffer:
            paragraphs.append(buffer)
        return paragraphs


if "__main__" == __name__:
    vector_db = MyVectorDB("demo")

    vector_db.add_documents("xdoc/llama2.pdf",
                            # page_numbers=[2, 3],
                            min_line_length=10)

    user_query = "llama2有多少参数"
    results = vector_db.search(user_query, 2)

    for para in results['documents'][0]:
        print(para + "\n")