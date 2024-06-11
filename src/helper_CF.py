from fastai.collab import CollabDataLoaders, collab_learner
import pandas as pd
import os
from fastai.vision.all import *
# import pathlib
from operator import itemgetter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_file(path):
    file_path_data = os.path.join(project_root, path)
    data_frame = pd.read_csv(file_path_data)
    return data_frame

# file_path_data = os.path.join(os.path.dirname(os.getcwd()), 'ModelAI/data/all_ratings.csv')

# all_ratings = read_file('src/data/all_ratings.csv')
all_ratings = read_file('data/all_ratings.csv')

# books = read_file('src/data/book-average-rating-with-langcode-eng-only.csv')
books = read_file('data/book-average-rating-with-langcode-eng-only.csv')


def load_learner_from_path(path_to_model):
    # file_path = os.path.join(os.path.dirname(os.getcwd()), 'ModelAI/training/fastai/all_ratings')
    data = CollabDataLoaders.from_df(all_ratings, seed=42, pct_val=0.3,
                                 item_name="description",rating_name='Book-Rating',
                                 user_name='User-ID')
    learn = collab_learner(data, use_nn = True, y_range=(0, 10.5)
                      #  ,loss_func=CrossEntropyLossFlat()
                      #  ,metrics=[accuracy, mse]
                       )
    learn = load_learner(path_to_model)
    
    return learn

def get_book_unrate(userId):
    rated_books = all_ratings[all_ratings['User-ID'] == userId]['description'].unique()
    all_books = all_ratings['description'].unique()
    unrated_books = set(all_books) - set(rated_books)
    # unrated_books_list = list(unrated_books)  # get id book
    unrated_books_info = books[books['description'].isin(unrated_books)]
    return unrated_books_info

def get_index_book_recommend(unrated_books_info):
    book_titles = unrated_books_info['isbn']
    top_books_list = list(book_titles)
    title_to_idx = {title: idx for idx, title in enumerate(top_books_list)}
    item_idxs = [title_to_idx[title] for title in top_books_list]
    filtered_arr = [x for x in item_idxs if x <= 48243]

    return filtered_arr

def get_weights(learn, index_list_book):
    book_weights = learn.model.embeds[1].weight[index_list_book]
    return book_weights

def recommend_books_list(book_weights, unrated_books_info):
    book_pca = book_weights.pca(3)
    fac0,fac1,fac2 = book_pca.t()
    book_comp = [(f, i) for f,i in zip(fac0, unrated_books_info['isbn'])]
    sorted_books = sorted(book_comp, key=itemgetter(0), reverse=True)[:20]
    recommend = [b for (a,b) in sorted_books]

    return recommend


