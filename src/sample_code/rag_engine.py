from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_ollama import ChatOllama # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain.chains import create_retrieval_chain # type: ignore


def build_retriever(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """PDF를 로드하고 FAISS retriever를 반환합니다."""
    print("📄 문서 로딩 중...")
    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    print(f"   → {len(split_docs)}개 청크로 분할 완료")

    print("🔢 임베딩 생성 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever()


def build_llm(model: str = "gemma3:4b"):
    """ChatOllama LLM 인스턴스를 반환합니다."""
    print("🤖 Ollama LLM 로드 중...")
    return ChatOllama(model=model)


def ask(question: str, prompt_template: str, llm, retriever) -> dict:
    """
    주어진 질문을 RAG 체인으로 실행하고 응답을 반환합니다.

    Returns:
        {"answer": str, "context": list[Document]}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    return chain.invoke({"input": question})
