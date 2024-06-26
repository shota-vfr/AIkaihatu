

import streamlit as st 

if 'embedding' not in st.session_state:
    st.session_state['embedding'] = None

if 'llm' not in st.session_state:
    st.session_state['llm'] = None
    
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
    

    
    
st.title('文章から検索')

with st.form('モデルの情報'):
    endpoint = st.text_input('Endpoint')
    api_key = st.text_input('API key')
    deployment_embedding = st.text_input('Deployment (埋め込みモデル)')
    deployment_summary = st.text_input('Deployment (要約時に使用)')
    
    
    submit = st.form_submit_button('送信')
    
    #挙動が見てみる
    
st.write(submit)

if submit == True:
    
    st.write(endpoint)
    st.write(api_key)
    st.write(deployment_embedding)
    st.write(deployment_summary)
    
    
    from langchain.embeddings import AzureOpenAIEmbeddings
    
    st.session_state['embedding'] = AzureOpenAIEmbeddings(
        openai_api_base = endpoint,
        openai_api_key = api_key,
        deployment = deployment_embedding
    )

    from langchain.chat_models import AzureChatOpenAI
    
    st.session_state['llm'] = AzureChatOpenAI(
        openai_api_version='2023-05-15',
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=deployment_summary
        
    )
    
    #chromadbを読み込む
    from langchain.vectorstores import Chroma
    st.session_state['vectorstore'] = Chroma(
        persist_directory = 'vectorstore',#データベースを置く場所
        embedding_function=st.session_state['embedding']#どの埋め込みモデルを使うか
    )
    
#ファイルのアップロード
uploaded_file = st.file_uploader('PDFのアップロード',type='pdf')

#PDFをフォルダに保存
from pathlib import Path
if uploaded_file is not None:
    save_path = Path('pdf_file',uploaded_file.name)
    with open(save_path,mode='wb')as w:
        w.write(uploaded_file.getvalue())
        
#データベースの読み込み確認
if st.session_state['vectorstore']is not None:
    
    #データベースに保存開始ボタン
    start_indexing = st.button("chromaDBに保存開始")
    
    #すでに保存さああれているファイルを確認
    files_in_db = list(set([file['source'] for file in st.session_state['vectorstore'].get()['metadatas']]))
    
    #ボタンを押したら保存開始
    if start_indexing:
            
        st.write('PDFから読み込み,,,')
        #フォルダ内のすべてのPDFファイル（保存済みを除く）
        from langchain.document_loaders import DirectoryLoader
        pdf_loader = DirectoryLoader('pdf_file',glob='*.pdf',exclude=files_in_db)
        documents = pdf_loader.load()
        
        st.write('読み込まれたドキュメント数：',len(documents))
        
        #ドキュメントが1つ以上読み込まれたら保存開始
        if len(documents):
            from langchain.text_splitter import CharacterTextSplitter
            text_splitter = CharacterTextSplitter(chunk_size=300,
                                                chunk_overlap=30)
            st.write('チャンキング中...')
            chunks = text_splitter.split_documents(documents)
            
            st.write('データベースに保存中...')
            st.session_state['vectorstore'].add_documents(chunks)
            st.session_state['vectorstore'].persist()
        
    #データベースに保存済みのファイルを確認
    files_in_db = list(set([file['source'] for file in st.session_state['vectorstore'].get()['metadatas']]))
    
    selected_files = st.multiselect('検索対象のファイルを選択',options=files_in_db)
    
    #文章から検索したい内容
    user_input = st.text_area('検索内容')
    
    start_search = st.button('検索')
    
    if start_search:
        # データベースの類似検索の条件指定
        retriever = st.session_state['vectorstore'].as_retriever(
        search_type='similarity',
        search_kwargs={"k":2, "filter":{'source':{'$in':selected_files}}}
        )

  # 類似検索開始(user_inputに類似するチャンクをみつける)
        retrieved_docs = retriever.invoke(user_input)

        # retrieved_docsを要約してユーザーに返したい
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
  # テンプレート
        template = '''
  必ず「検索結果」を基に、「ユーザーからの質問」に答えなさい。
  出力は質問への回答の要約のみを含むこと。

  【ユーザーからの質問】
  {input1}

  【検索結果】
  {input2}
  '''

        prompt_template = PromptTemplate(
        input_variables=['input1', 'input2'],
        template=template
  )

  # テンプレートとLLMを連結
        chain = LLMChain(llm=st.session_state['llm'], prompt=prompt_template)

  # 検索結果を要約
        extracted_texts = '\n\n'.join([doc.page_content for doc in retrieved_docs])
        ans = chain.run(input1=user_input, input2=extracted_texts)

        st.write(ans)