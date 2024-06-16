import streamlit as st
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'
MODEL_PATH = r'C:\Users\91638\Desktop\Training\Data Analytics\Projects\CSVWhiz\llama-2-7b-chat.ggmlv3.q4_0.bin'

# Define available models for classification
classification_models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machines (SVM)": SVC(),
    "k-Nearest Neighbors (k-NN)": KNeighborsClassifier()
}

# Define available models for regression
regression_models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Support Vector Regression (SVR)": SVR()
}

@st.cache_resource
def load_llm(model_path):
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def train_classification_models(X_train, X_test, y_train, y_test):
    results = {}
    for model_name, model in classification_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {"model": model, "accuracy": accuracy}
    return results

def train_regression_models(X_train, X_test, y_train, y_test):
    results = {}
    for model_name, model in regression_models.items():
        if model_name == "Polynomial Regression":
            poly_features = PolynomialFeatures(degree=2)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        if "Regression" in model_name:
            mse = mean_squared_error(y_test, y_pred)
            results[model_name] = {"model": model, "mse": mse}
        else:
            mse = None
            results[model_name] = {"model": model, "y_pred": y_pred}
    return results

def preprocess_data(X_train, X_test, scale_option):
    numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Define preprocessing steps
    if scale_option == "Standard Scaler":
        scaler = StandardScaler()
    elif scale_option == "Min-Max Scaler":
        scaler = MinMaxScaler()
    elif scale_option == "Robust Scaler":
        scaler = RobustScaler()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', scaler)  # Scale features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed

def plot_pairplot(df):
    st.write("Pairplot:")
    fig = sns.pairplot(df)
    st.pyplot(fig)

def plot_heatmap(df):
    st.write("Correlation Heatmap:")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot()

def plot_histogram(df):
    st.write("Histogram:")
    fig, ax = plt.subplots()
    for col in df.columns:
        sns.histplot(df[col], ax=ax, kde=True)
    st.pyplot(fig)

def plot_scatterplot(df):
    st.write("Scatter Plot:")
    columns = list(df.columns)
    x_axis = st.selectbox("Select X-axis:", options=columns)
    y_axis = st.selectbox("Select Y-axis:", options=columns)
    sns.scatterplot(x=df[x_axis], y=df[y_axis])
    st.pyplot()

def plot_categorical_barplot(df):
    st.write("Species Count:")
    sns.countplot(x='species', data=df)
    st.pyplot()

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

    st.sidebar.subheader("Modeling Options")
    target = st.sidebar.selectbox("Select Target Variable", options=df.columns)
    target_type = st.sidebar.selectbox("Select Problem Type", options=["classification", "regression"])

    scale_data = st.sidebar.checkbox("Scale Data")
    if scale_data:
        scale_option = st.sidebar.selectbox("Select Scaling Method", ["Standard Scaler", "Min-Max Scaler", "Robust Scaler"])
    else:
        scale_option = None

    st.sidebar.subheader("Visualization Options")
    plot_options = {
        "Pairplot": plot_pairplot,
        "Heatmap": plot_heatmap,
        "Histogram": plot_histogram,
        "Scatter Plot": plot_scatterplot,
        "Species Count": plot_categorical_barplot  # Added categorical plot option
    }
    default_plot = "Pairplot"
    selected_plot = st.sidebar.selectbox("Select Plot Type", options=list(plot_options.keys()), index=0)

    if st.sidebar.button("Run Modeling"):
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if scale_data and scale_option:
            X_train, X_test = preprocess_data(X_train, X_test, scale_option)

        if target_type == "classification":
            with st.spinner("Training classification models..."):
                results = train_classification_models(X_train, X_test, y_train, y_test)
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            best_model = results[best_model_name]
            accuracy_percentage = best_model['accuracy'] * 100
            st.subheader(f"Best Classification Model: {best_model_name}")
            st.write(f"Accuracy: {accuracy_percentage:.2f}%")
        elif target_type == "regression":
                with st.spinner("Training regression models..."):
                    results = train_regression_models(X_train, X_test, y_train, y_test)
                best_model_name = min(results, key=lambda x: results[x]['mse'])
                best_model = results[best_model_name]
                st.subheader(f"Best Regression Model: {best_model_name}")
                st.write(f"Mean Squared Error: {best_model['mse']:.2f}")

    st.subheader("Data Visualizations")

    # Create a layout with columns for the selected plot and default pairplot
    plot_col, pairplot_col = st.columns(2)

    # Execute the selected plot function
    plot_function = plot_options[selected_plot]
    with plot_col:
        plot_function(df)

    # Display the default pairplot in the other column if a different plot is selected
    with pairplot_col:
        if selected_plot != "Pairplot":
            plot_pairplot(df)

    # Chat with CSV Data
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
