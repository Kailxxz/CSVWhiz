import streamlit as st
import tempfile
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from tpot import TPOTClassifier, TPOTRegressor
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'
MODEL_PATH = r'C:\Users\91638\Desktop\Training\Data Analytics\Projects\CSVWhiz\llama-2-7b-chat.ggmlv3.q4_0.bin'

@st.cache_resource
def load_llm(model_path):
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def preprocess_data(X_train, X_test, scale_option):
    numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if scale_option == "Standard Scaler":
        scaler = StandardScaler()
    elif scale_option == "Min-Max Scaler":
        scaler = MinMaxScaler()
    elif scale_option == "Robust Scaler":
        scaler = RobustScaler()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed

st.title("CSVWhiz")
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    df = pd.read_csv(tmp_file_path)

    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH) 

    llm = load_llm(MODEL_PATH)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        if query.lower().strip() in ["hello", "hi", "csv whiz", "csv file"]:
            uploaded_filename = uploaded_file.name if uploaded_file else "the uploaded file"
            response = f"Hello! I'm CSV Whiz, here to assist you with any questions related to {uploaded_filename}."
        else:
            chat_history = [(message[0], message[1]) for message in st.session_state['history']]
            result = chain({"question": query, "chat_history": chat_history})
            response = result["answer"]
        st.session_state['history'].append((query, response))
        return response

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    st.subheader("Data Insights")
    st.write("Data Preview:")
    st.dataframe(df.head())

    st.write("Data Summary:")
    st.write(df.describe())

    st.write("Pairplot:")
    fig = sns.pairplot(df)
    st.pyplot(fig)

    st.sidebar.subheader("Modeling Options")
    target = st.sidebar.selectbox("Select Target Variable", options=df.columns)
    target_type = st.sidebar.selectbox("Select Problem Type", options=["classification", "regression"])
    scale_data = st.sidebar.checkbox("Scale Data")
    if scale_data:
        scale_option = st.sidebar.selectbox("Select Scaling Method", ["Standard Scaler", "Min-Max Scaler", "Robust Scaler"])
    else:
        scale_option = None

    if st.sidebar.button("Run Modeling"):
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if scale_data and scale_option:
            X_train, X_test = preprocess_data(X_train, X_test, scale_option)

        if target_type == "classification":
            with st.spinner("Running AutoML for classification..."):
                tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
                tpot.fit(X_train, y_train)
                accuracy = tpot.score(X_test, y_test)
                st.subheader("TPOT Classification Results")
                st.write(f"Best Model Accuracy: {accuracy:.2f}")
                tpot.export('tpot_best_classification_pipeline.py')
        elif target_type == "regression":
            with st.spinner("Running AutoML for regression..."):
                tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
                tpot.fit(X_train, y_train)
                mse = tpot.score(X_test, y_test)
                st.subheader("TPOT Regression Results")
                st.write(f"Best Model Mean Squared Error: {mse:.2f}")
                tpot.export('tpot_best_regression_pipeline.py')

    st.subheader("Chat with CSV Data")

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("You:", placeholder="Type your message here", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                with st.spinner("Processing..."):
                    output = conversational_chat(user_input)

    with response_container:
        if 'history' in st.session_state:
            for i, (user_msg, bot_msg) in enumerate(st.session_state['history']):
                st.write(f"You: {user_msg}", unsafe_allow_html=True)
                st.write(f"Bot: {bot_msg}", unsafe_allow_html=True)
