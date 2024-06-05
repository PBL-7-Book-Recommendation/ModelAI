from flask import Flask, request, jsonify
from helper import *
import pathlib
import os

app = Flask(__name__)
pathlib.PosixPath = pathlib.WindowsPath
path_to_model = os.path.join(os.path.dirname(os.getcwd()), 'src/models/all_ratings_fit_30_5.pkl')
learn = load_learner_from_path(path_to_model)

@app.route('/recommend_book/<userID>', methods=['GET'])
def recommend_books_for_user(userID):
    unrated_books_info = get_book_unrate(userID)
    index_list_book = get_index_book_recommend(unrated_books_info)
    book_weights = get_weights(learn, index_list_book)
    recommend_books = recommend_books_list(book_weights, unrated_books_info)

    return jsonify({"recommend_books": recommend_books})

@app.route('/', methods=['GET'])
def default_router():
    a = 'hello'
    return a

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)