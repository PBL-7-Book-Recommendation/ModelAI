import numpy as np

# Content-based recommendation
def get_recommended_book_ids(book_id, cos_sim):
    # find the index of the book
    book_index = np.where(cos_sim[:, 0, 0] == book_id)
    if len(book_index[0]) == 0:
        return []
    # get the top 10 similar books
    recommended_books = cos_sim[book_index, 1:11, 0][0][0]
    return list(recommended_books)
