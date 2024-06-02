# CSVWhiz Bot

CSVWhiz Bot is an interactive tool that allows users to interact with CSV files using natural language. It leverages the power of Llama 2, Sentence Transformers, CTransformers, Langchain, and Streamlit to provide a seamless and intuitive user experience.

## Features

- **Natural Language Interaction:** Use natural language queries to extract information from CSV files.
- **Powerful NLP Models:** Leverage Llama 2 for language understanding and Sentence Transformers for sentence embedding.
- **Efficient Transformers:** Utilize CTransformers for optimized model inference.
- **User-Friendly Interface:** A Streamlit-based UI for easy interaction.

## Installation

Follow these steps to set up the CSVWhiz Bot on your local machine:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/CSVWhiz.git
    cd CSVWhiz
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your CSV file:**
   Ensure your CSV file is formatted correctly and saved in an accessible location.

2. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

## Project Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Technologies Used

- **Llama 2:** Advanced language model for natural language understanding.
- **Sentence Transformers:** Efficient sentence embedding for semantic similarity.
- **CTransformers:** Optimized transformer inference.
- **Langchain:** Chain of language models for complex NLP tasks.
- **Streamlit:** Fast way to build and share data apps.

Happy querying with CSVWhiz Bot!
