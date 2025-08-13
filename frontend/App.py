import streamlit as st
import requests

# Specify the URL name 
API_URL = r"http://localhost:8000//relevent_docs"       # Give your external url to host the model

st.title("Information Retrival App")

st.markdown("Enter the details")

segmenter = st.selectbox("Sentece segmentation technique", options=["naive", "punkt"])
tokenizer = st.selectbox("okenization technique", options=["naive", "ptb"])
method = st.selectbox("Information Retrival techniques", options=["tfidf", "lsa", "esa"])
query = st.text_input("Enter the query", value="what papers are avalable on the buckling of emty cylindrical shells")

if st.button("Search Relevent document Ids"):
    input_data = {
            "segmenter" : segmenter,
            "tokenizer" : tokenizer,
            "method" : method,
            "query" : query
        }
    
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            display = f"Original query : {result['original_query']} \nCorrected query : {result['corrected_query']} \nRelevent Ids : {result['doc_ids']}"
            st.success(f"Result : **{display}**")
        else:
            st.error(f"API error : {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to fastapi server. Make sure it's running on 8000 port")













# from flask import Flask, render_template, request, redirect, url_for
# from backend.main import get_top_doc_ids  # Import the function

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def welcome():
#     return redirect(url_for('query_submit', message="Welcome to my page"))

# @app.route("/query_submit/<message>", methods=['GET', 'POST'])
# def query_submit(message):
#     if request.method == 'POST':
#         query = request.form['query']
#         return redirect(url_for('show_relevant_ids', query=query))
#     return render_template('query_info.html', message=message)

# @app.route('/show_relevant_ids/<query>', methods=['GET', 'POST'])
# def show_relevant_ids(query):
#     corrected_query, doc_ids = get_top_doc_ids(query)
#     return render_template('results.html', query=corrected_query, doc_ids=doc_ids)

# if __name__ == "__main__":
#     app.run(debug=True)
