from dotenv import load_dotenv
load_dotenv()

# app.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# LLM呼び出し関数（課題要件）
# -----------------------------
def run_llm(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、
    LLMの回答を文字列として返す
    """

    # 専門家ごとのシステムメッセージ
    if expert_type == "A":
        system_message = (
            "あなたは経験豊富なPythonプログラミングの専門家です。"
            "初心者にも分かりやすく、具体例を交えて説明してください。"
        )
    else:
        system_message = (
            "あなたは優秀なビジネスコンサルタントです。"
            "経営視点・実務視点で、簡潔かつ論理的に説明してください。"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    chain = prompt | llm
    result = chain.invoke({"input": user_input})

    return result.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("LLM搭載 Webアプリ（LangChain × Python）")

st.markdown("""
### アプリ概要
このアプリは、入力したテキストをもとに  
**選択した専門家の立場でLLMが回答を生成するWebアプリ**です。

### 操作方法
1. 専門家の種類をラジオボタンで選択してください  
2. 質問や相談内容を入力してください  
3. 「送信」ボタンを押すと、LLMの回答が表示されます
""")

expert = st.radio(
    "専門家の種類を選択してください",
    options=["A", "B"],
    format_func=lambda x: "Pythonの専門家" if x == "A" else "ビジネスコンサルタント"
)

user_input = st.text_area("質問・相談内容を入力してください")

if st.button("送信"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("LLMが回答を生成中..."):
            answer = run_llm(user_input, expert)
        st.subheader("💡 回答")
        st.write(answer)
