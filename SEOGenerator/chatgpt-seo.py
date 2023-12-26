import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

st.header("SEO Article Generator")
def generate_articles(keyword: str, style: str, word_count: int) -> str:
    response =  client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {"role": "user", "content": "Write a SEO optimized article about " + keyword},
            {"role": "user", "content": "Keep the style as " + style},
            {"role": "user", "content": "Keep the word count to less than " + str(word_count)}
        ]
    )

    result = ""
    for choice in response.choices:
        result += choice.message.content
    return result


keyword = st.text_input("Enter a keyword")
style = st.selectbox("Select a writing style", ["Funny", "academic", "Poetic"])
word_count = st.slider("Word Count", min_value=100, max_value=1000, value=300)
submit_button = st.button("Generate Article")

if submit_button:
    message = st.empty()
    message.text = "Generating article ..."
    article = generate_articles(keyword, style, word_count)
    message.text = ""
    st.write(article)
    st.download_button(
        label="Download Article",
        data=article,
        file_name="article.text",
        mime="text/plain",
    )
