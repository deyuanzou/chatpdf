"""
@author: yuan
@date: 2023/12/09 13:13
@intro: 使用langchain连接gpt-3.5-turbo大语言模型对上传的PDF文件进行处理
"""
import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms.openai import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1.设置UI界面，对项目进行描述
with st.sidebar:
    st.title("Ask for PDF")
    st.markdown("""
    本项目提供PDF文件上传的功能，
    你可以输入问题，模型会在PDF文件中寻找答案
    """)
    add_vertical_space(5)
    st.write("开始你的大模型之旅")


# 2.生成main函数作为程序的入口
def main():
    # 加载环境变量
    api_key = st.text_input("输入你的OpenAI API Key")
    if not api_key:
        st.error("请先输入API Key")
        return
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://flag.smarttrot.com/v1"
    st.header("Ask for PDF")
    # 3.上传pdf文件
    pdf = st.file_uploader("请上传PDF文件", type="pdf")
    if pdf:
        st.write(pdf.name)
    else:
        st.write("你还没有上传PDF文件")
        print("用户未上传PDF文件")
        return
    # 读取pdf文件
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    st.write(text)
    # 4.对文本进行切割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # chunk_size指定切割文件块的大小
        chunk_overlap=20,  # chunk_overlap指定相邻文件块之间重叠的字符数
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    st.write(chunks)
    store_name = pdf.name[:-4]
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vectorStore = pickle.load(f)
        st.write("嵌入已经从磁盘加载")
    else:
        # 5.将切割后的文件进行向量化，存储到向量数据库中
        embeddings = OpenAIEmbeddings()
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        # 6.将向量数据库持久化
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorStore, f)
        st.write("向量嵌入完成")

    # 7.用户输入，大模型搜索向量数据库，返回结果
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    query = st.text_input("请输入与上传的PDF文件相关的问题")
    if query:
        docs = vectorStore.similarity_search(query=query, k=1)
        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)


if __name__ == '__main__':
    main()
