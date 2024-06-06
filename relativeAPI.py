from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import os
from src.helper_CF import *
from src.helper_CB import *
import pathlib
from src.database.config import *


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
learn = load_learner_from_path(path_to_model_CF)

# defaul api
@app.route('/', methods=['GET'])
def default_router():
    a = 'hello'
    return a


# API để trả về danh sách sách dựa trên danh sách id
@app.route("/content-based-recommend/<string:book_id>", methods=["GET"])
@cross_origin(origin="*")  # Fix to current web domain
def get_books(book_id):
    if not book_id:
        return jsonify({"error": "No book ID provided"}), 400

    recommend_books = get_recommended_book_ids(book_id, cos_sim=cos_sim)
    print(recommend_books)

    # Truy vấn thông tin sách và tác giả từ cơ sở dữ liệu
    books = query_db(books_query, (recommend_books,))

    # Truy vấn thông tin nguồn từ cơ sở dữ liệu
    source_ids = [book["source_id"] for book in books]

    sources = {
        source["id"]: source for source in query_db(sources_query, (source_ids,))
    }

    # Truy vấn thông tin tương tác từ cơ sở dữ liệu
    interactions = {
        interaction["book_id"]: interaction["interactions"]
        for interaction in query_db(interactions_query, (recommend_books,))
    }

    # Kết hợp thông tin nguồn và interactions vào dữ liệu sách
    for book in books:
        book["source"] = sources.get(
            book["source_id"], {"id": book["source_id"], "name": "Unknown"}
        )
        book["interactions"] = interactions.get(book["id"], [])

    return jsonify({"data": books})


@app.route('/recommend_CF_book/<userID>', methods=['GET'])
@cross_origin(origin="*") 
def recommend_books_for_user(userID):
    unrated_books_info = get_book_unrate(userID)
    index_list_book = get_index_book_recommend(unrated_books_info)
    book_weights = get_weights(learn, index_list_book)
    recommend_books = recommend_books_list(book_weights, unrated_books_info)

    return jsonify({"recommend_books": recommend_books})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
