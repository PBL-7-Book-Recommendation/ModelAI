from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import os
from src.helper_CF import *
from src.helper_query import *


app = Flask(__name__)
# Apply Flask CORS
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# path_to_model_CF = os.path.join(os.path.dirname(os.getcwd()), './src/models/all_ratings_fit_30_5.pkl')
path_to_model_CF = os.path.join(project_root, "models/all_ratings_11_6.pth")


# defaul api
@app.route('/', methods=['GET'])
@cross_origin(origin="*") 
def default_router():
    print(project_root,'project_root')

    a = 'hello'
    return a

learn = load_learner_from_path(path_to_model_CF)
@app.route('/recommend_CF_book/<userID>', methods=['GET'])
@cross_origin(origin="*") 
def recommend_books_for_user(userID):
    bookrated =  get_interactions_by_userid(userID)
    recommend_books = recommend_books_list(learn, bookrated)
    print(recommend_books,'recom cf',userID,'id')
    recommend_books_CF = get_books_by_ids(recommend_books)

    return jsonify({"data": recommend_books_CF})

    # return jsonify({"recommend_books": recommend_books})

if __name__ == "__main__":

    # port = int(os.environ.get("PORT", 5000)) 
    port = os.getenv('FLASK_PORT', '5000')
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=True)
