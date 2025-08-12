import os, json, tempfile
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader # PDF 로더
from langchain.text_splitter import RecursiveCharacterTextSplitter # 텍스트 분할기
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# --- 함수 정의 ---

# 1) PDF 파일 처리 함수 (신규 추가)
def process_pdf(uploaded_file):
    # Streamlit의 UploadedFile 객체를 임시 파일로 저장하여 경로를 얻습니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # PyPDFLoader를 사용하여 PDF 문서 로드
    loader = PyPDFLoader(tmp_file_path)
    docs_from_pdf = loader.load()

    # 임시 파일 삭제
    os.remove(tmp_file_path)

    # 텍스트 분할기(Splitter)를 사용하여 문서를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs_from_pdf)
    return split_docs

# 2) 앙상블 Retriever 구성(BM25 + FAISS)
# @st.cache_data # 리소스가 큰 경우 캐싱하여 성능 향상
def build_ensemble_retriever(docs):
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas); bm25.k = 2
    embedding = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    faiss_store = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    faiss = faiss_store.as_retriever(search_kwargs={"k": 2})
    return EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.3, 0.7])

# 3) OpenAI LLM 초기화(캐시)
@st.cache_resource(show_spinner=False)
def load_openai_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. Streamlit Secrets 또는 환경변수에 설정하세요.")
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# 4) 검색 함수(문서 리스트 반환)
def search(query: str, retriever):
    return retriever.invoke(query)

# 5) RAG 프롬프트 구성
def build_prompt(query: str, docs):
    lines = ["아래 '자료'만 근거로 한국어로 간결히 답하세요.",
             "- 자료 밖 정보를 추측하지 마세요.",
             "- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.\n",
             f"질문:\n{query}\n",
             "자료:"]
    for i, d in enumerate(docs, 1):
        lines.append(f"[문서{i}] (source: page {d.metadata.get('page', 'N/A')})\n{d.page_content}\n")
    lines.append("답변:")
    return "\n".join(lines)

# 6) OpenAI로 답변 생성
def generate_with_llm(llm: ChatOpenAI, prompt: str) -> str:
    resp = llm.invoke(prompt)
    return resp.content.strip()

# --- Streamlit UI 구성 ---
def main():
    st.set_page_config(page_title="🤖 ChatPDF", page_icon="📄", layout="centered")
    st.title("🤖 ChatPDF")
    st.markdown("---")

    # LLM 모델 초기화
    try:
        llm = load_openai_llm()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # 사이드바에 PDF 업로더 추가
    with st.sidebar:
        st.header("PDF 파일 업로드")
        uploaded_file = st.file_uploader("여기에 PDF를 드롭하거나 클릭하여 업로드하세요.", type="pdf")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # 파일이 업로드 되었을 때만 처리
    if uploaded_file:
        #이전에 처리한 파일과 다른 파일이면 새로 처리
        if st.session_state.get("processed_file_name") != uploaded_file.name:
            with st.spinner("PDF 파일을 처리 중입니다... 잠시만 기다려주세요."):
                docs = process_pdf(uploaded_file)
                st.session_state.retriever = build_ensemble_retriever(docs)
                st.session_state.processed_file_name = uploaded_file.name
                st.session_state.messages = [] # 새 파일이므로 대화 기록 초기화
            st.success(f"'{uploaded_file.name}' 파일 처리가 완료되었습니다! 이제 질문을 시작하세요.")

    # 대화 기록 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력 처리
    user_input = st.chat_input("PDF 내용에 대해 질문해보세요.")
    if user_input:
        if not st.session_state.retriever:
            st.error("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")
            st.stop()

        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("답변을 생성 중입니다..."):
            docs = search(user_input.strip(), st.session_state.retriever)

            if not docs:
                answer = "업로드된 PDF 문서에서 관련 내용을 찾지 못했습니다."
            else:
                prompt = build_prompt(user_input.strip(), docs)
                answer = generate_with_llm(llm, prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("🔎 사용한 자료(검색 결과) 보기", expanded=False):
            if not docs:
                st.markdown("_검색 결과 없음_")
            else:
                for d in docs:
                    st.markdown(f"**[Page {d.metadata.get('page', 'N/A')}]**\n\n{d.page_content}")
    
    elif not uploaded_file:
        st.info("왼쪽 사이드바에서 PDF 파일을 업로드하여 대화를 시작하세요.")

# --- 앱 엔트리포인트 ---
if __name__ == "__main__":
    main()