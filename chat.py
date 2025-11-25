import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil # shutil 모듈을 임포트해야 합니다 (코드 상단에 추가 필요)
import os

# ----------------------------------------------------
# --- 모델 및 경로 설정 (수정됨) ---
# ----------------------------------------------------
# ✅ 수정: 벡터 DB 파일이 저장된 저장소 ID만 지정합니다.
HF_REPO_ID = "ju03/Healthcare_knowledge_chatbot" 

VECTOR_DB_LOCAL_PATH = os.path.join(os.getcwd(), "vector_db")

# ✅ 수정: LLM 모델은 공개된 KoAlpaca 모델 ID로 지정하여 로딩 문제를 해결합니다.
LLM_MODEL_PATH = "Beomi/KoAlpaca-Polyglot-12.8B" 

EMBEDDING_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# ----------------------------------------------------


# 🔑 누락된 함수 정의: HuggingFacePipeline의 출력을 처리하는 람다 함수
def extract_hf_output_text(output):
    """HuggingFacePipeline의 출력이 리스트/딕셔너리 형태일 경우 텍스트만 추출합니다."""
    if isinstance(output, list) and output and isinstance(output[0], dict) and 'generated_text' in output[0]:
        return output[0]['generated_text']
    if isinstance(output, str):
        return output
    return str(output)


# 🔑 코사인 유사도 계산 함수 정의
def calculate_similarity(embedding_function, answer, ground_truth):
    """답변과 정답 간의 코사인 유사도를 계산합니다."""
    if not ground_truth or not answer:
        return 0.0

    try:
        embeddings_list = embedding_function.embed_documents([answer, ground_truth])
    except Exception as e:
        print(f"임베딩 계산 오류: {e}")
        return 0.0

    answer_embedding = np.array(embeddings_list[0]).reshape(1, -1)
    gt_embedding = np.array(embeddings_list[1]).reshape(1, -1)

    similarity = cosine_similarity(answer_embedding, gt_embedding)[0][0]
    return similarity


@st.cache_resource
def load_rag_pipeline():
    """
    RAG 파이프라인의 모든 구성 요소 (LLM, Embeddings, Retriever, QA Chain)를 로드합니다.
    """
    llm_obj = None
    retriever = None
    embeddings = None

    # status = st.status("**:gear: RAG 챗봇 구성 요소를 로드 중입니다...**", expanded=True)

    try:
        # 1. FAISS Vector DB 파일 다운로드 및 준비

        os.makedirs(VECTOR_DB_LOCAL_PATH, exist_ok=True)

        faiss_filename_in_repo = "vector_db/index.faiss"
  
        pkl_filename_in_repo = "vector_db/index.pkl"


        # index.faiss와 index.pkl 다운로드
        downloaded_faiss_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=faiss_filename_in_repo,
            # local_dir=VECTOR_DB_LOCAL_PATH,
            local_dir_use_symlinks=False
        )
        downloaded_pkl_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=pkl_filename_in_repo,
            # local_dir=VECTOR_DB_LOCAL_PATH,
            local_dir_use_symlinks=False
        )
        final_faiss_path = os.path.join(VECTOR_DB_LOCAL_PATH, "index.faiss")
        final_pkl_path = os.path.join(VECTOR_DB_LOCAL_PATH, "index.pkl")
        
        # 파일을 최종 위치로 이동
        shutil.copy(downloaded_faiss_path, final_faiss_path)
        shutil.copy(downloaded_pkl_path, final_pkl_path)

        
        # 2. Embeddings 모델 로드
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        

        # 3. Vector Store 로드 및 Retriever 생성
        vectorstore = FAISS.load_local(
            folder_path=VECTOR_DB_LOCAL_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        


        # 4. LLM 객체 로드 및 HuggingFacePipeline 생성 (답변 생성 모델)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        llm_obj = HuggingFacePipeline(pipeline=pipe)


        # 5. Retrieval QA Chain 구성
        custom_prompt = PromptTemplate.from_template(
             """[지시]: 제공된 "문맥(Context)"만을 사용하여 "질문(Question)"에 한국어로 답변하세요.
             정보가 없다면, "제공된 정보 내에서는 답변할 수 없습니다."라고 응답하세요.
             **오직 최종 답변 내용만 출력하세요. 다른 어떤 형식이나 추가 문구도 사용하지 마세요.**

             문맥(Context): {context}

             질문(Question): {question}

             답변:"""
        )

        rag_answer_chain = (
            custom_prompt
            | llm_obj
            | RunnableLambda(extract_hf_output_text)
            | StrOutputParser()
        )

        qa_chain = RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever,
        ).assign(
            answer=rag_answer_chain
        )
        status.success(" 최종 구성 완료!")
        status.update(label="**:white_check_mark: RAG 챗봇 로드 완료!**", state="complete", expanded=False)
        return qa_chain, retriever, embeddings

    except Exception as e:
        if status :
            status.error(f"❌ **RAG 파이프라인 로드 실패!** 오류 상세: {e}")
        else:
            st.error(f"❌ **RAG 파이프라인 로드 실패!** 오류 상세: {e}")
        return None, None, None


# --- Streamlit UI 시작 ---

st.set_page_config(page_title="💖 한국어 헬스케어 챗봇", layout="wide")
st.title('🩺 한국어 건강 정보 RAG 챗봇')
st.caption(f'Hugging Face 리포지토리: **{HF_REPO_ID}**')

# 챗봇 구성 요소 로드
qa_chain, retriever, embeddings = load_rag_pipeline()

if qa_chain is None:
    st.warning("챗봇 기능이 비활성화되었습니다. 위 오류 메시지를 확인하세요.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("건강에 대해 궁금한 점을 물어보세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                try:
                    # RAG 체인 실행
                    result = qa_chain.invoke({"question": prompt})
                    
                    response = result['answer'].strip() # LLM 답변
                    # ✅ 수정: source_docs를 context 키에서 가져옵니다.
                    source_docs = result['context']     
                    
                    # ⚠️ 개선된 답변 프리픽스 처리: '답변:'이 포함된 경우 제거
                    response = response.strip()
                    if "답변:" in response:
                        response = response.split("답변:", 1)[1].strip()
                    elif response.lower().startswith("답변"):
                        response = response[len("답변"):].strip()

                    final_output = response
                    
                    similarity_text = ""
                    if source_docs:
                        # 검색된 청크 내용을 합쳐서 "정답 대리" 텍스트 생성
                        retrieved_text = " ".join([doc.page_content for doc in source_docs])
                        
                        # 코사인 유사도 계산
                        similarity_score = calculate_similarity(embeddings, final_output, retrieved_text)
                        
                        similarity_text = f"\n\n---"
                        similarity_text += f"\n**📝 답변 품질 평가 (검색된 문헌 대비 유사도):** `{similarity_score:.4f}`"
                        similarity_text += f" (1에 가까울수록 문헌 내용을 잘 반영)"
                    else:
                        similarity_text = "\n\n---"
                        similarity_text += "\n**⚠️ 답변 품질 평가:** 검색된 문헌이 없어 유사도를 측정할 수 없습니다."
                        
                    
                    st.markdown(final_output + similarity_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_output + similarity_text})

                except Exception as e:
                    st.error(f"**답변 생성 중 심각한 오류 발생:**")
                    st.exception(e)
