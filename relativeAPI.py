from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import os
from src.helper_CF import *
from src.helper_CB import *
from src.helper_query import *
import pathlib

app = Flask(__name__)
# Apply Flask CORS
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

cos_sim = np.load(
    "./data/top_similar_books.npy",
    allow_pickle=True,
)

pathlib.PosixPath = pathlib.WindowsPath

path_to_model_CF = os.path.join(os.path.dirname(os.getcwd()), './src/models/all_ratings_fit_30_5.pkl')
# path_to_model_CF = os.path.join(os.path.dirname(os.getcwd()), './ModelAI/models/all_ratings_fit_30_5.pkl')

learn = load_learner_from_path(path_to_model_CF)

# Default API
@app.route('/', methods=['GET'])
def default_router():
    return 'hello'

# API to return book list based on book ID
@app.route("/content-based-recommend/<string:book_id>", methods=["GET"])
@cross_origin(origin="*")  # Fix to current web domain
def get_books(book_id):
    if not book_id:
        return jsonify({"error": "No book ID provided"}), 400

    recommend_books = get_recommended_book_ids(book_id, cos_sim=cos_sim)
    print(recommend_books)

    recommend_books_CB = get_books_by_ids(recommend_books)

    return jsonify({"data": recommend_books_CB})

@app.route('/recommend_CF_book/<userID>', methods=['GET'])
@cross_origin(origin="*")
def recommend_books_for_user(userID):
    unrated_books_info = get_book_unrate(userID)
    index_list_book = get_index_book_recommend(unrated_books_info)
    book_weights = get_weights(learn, index_list_book)
    recommend_books = recommend_books_list(book_weights, unrated_books_info)
    print(recommend_books, 'recom cf')
    recommend_books_CF = get_books_by_ids(recommend_books)

    return jsonify({"data": recommend_books_CF})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT environment variable if available
    print(port, 'port')
    app.run(host="0.0.0.0", port=port, debug=True)
