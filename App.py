from flask import Flask, render_template, request, redirect, url_for
from search_engine_runner import get_top_doc_ids  # Import the function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return redirect(url_for('query_submit', message="Welcome to my page"))

@app.route("/query_submit/<message>", methods=['GET', 'POST'])
def query_submit(message):
    if request.method == 'POST':
        query = request.form['query']
        return redirect(url_for('show_relevant_ids', query=query))
    return render_template('query_info.html', message=message)

@app.route('/show_relevant_ids/<query>', methods=['GET', 'POST'])
def show_relevant_ids(query):
    corrected_query, doc_ids = get_top_doc_ids(query)
    return render_template('results.html', query=corrected_query, doc_ids=doc_ids)

if __name__ == "__main__":
    app.run(debug=True)
