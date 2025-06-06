import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import numpy as np
from transformers import pipeline
# from dotenv import load_dotenv

import gradio as gr

# load_dotenv()

try:
    books = pd.read_csv('books_with_categories.csv')
    books["large_thumbnail"] = np.where(
        books["thumbnail"].notna(),
        books["thumbnail"] + "&fife=w800",
        "cover-not-found.jpg"
    )
    categories = ["All"] + sorted(books["simple_categories"].unique())
except Exception as e:
    print(f"Error loading book data: {e}")
    books = pd.DataFrame()
    categories = ["All"]



# Global variable for Chroma DB
db_books = None
persist_directory = 'db'

def set_openai_key(api_key):
    """Set OpenAI key and initialize ChromaDB"""
    if not api_key.strip():
        return "❌ Please enter a valid API key"
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        global db_books
        # Initialize Chroma DB with the provided API key
        db_books = Chroma(
            collection_name='v_db',
            persist_directory='db',
            embedding_function=OpenAIEmbeddings()
        )
        return "✅ API key set! You can now search for books"
    except Exception as e:
        return f"❌ Error initializing database: {str(e)}"



# db_books = Chroma(collection_name='v_db', persist_directory=persist_directory, embedding_function= OpenAIEmbeddings())


def retreive_semantic_recommendations(query: str, category:str = None, initial_topk: int = 50, 
                                      final_topk:int = 16) -> pd.DataFrame:
    
    if db_books is None:
        raise gr.Error("Database not initialized. Please set your OpenAI API key first!")
    
    recs = db_books.similarity_search(query, k = initial_topk)
    
    
    books_list = [int(rec.page_content.strip('"').split()[0].strip()) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_topk)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_topk)
    else:
        book_recs = book_recs.head(final_topk)

    return book_recs

def recommend_books(
        query: str,
        category: str,
):
    recommendations = retreive_semantic_recommendations(query, category)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")
    gr.Markdown("Find books similar to your description using AI-powered semantic search")
    
    # API Key Section
    with gr.Row():
        api_key = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            placeholder="sk-...",
            info="Get your key from [platform.openai.com](https://platform.openai.com/api-keys)"
        )
        key_btn = gr.Button("Set Key", variant="primary")
        key_status = gr.Textbox(label="Status", interactive=False)
    
    key_btn.click(set_openai_key, inputs=api_key, outputs=key_status)
    
    # Separator
    gr.Markdown("---")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A book on World War 2")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        # tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()