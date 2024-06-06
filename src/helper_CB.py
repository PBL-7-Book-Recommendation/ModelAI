import numpy as np

books_query = """
    SELECT
        b.id,
        b.title,
        b.book_cover AS "bookCover",
        b.language,
        b.image_url AS "imageUrl",
        b.release_date AS "releaseDate",
        b.price,
        b.average_rating AS "averageRating",
        b.source_id,
        json_agg(
            json_build_object(
                'author', json_build_object(
                    'id', a.id,
                    'name', a.name,
                    'avatar', a.avatar
                )
            )
        ) AS authors
    FROM book b
    LEFT JOIN author_to_book atb ON b.id = atb.book_id
    LEFT JOIN author a ON atb.author_id = a.id
    WHERE b.id = ANY(%s)
    GROUP BY b.id
"""

sources_query = """
    SELECT id, name
    FROM source
    WHERE id = ANY(%s)
"""

interactions_query = """
    SELECT
        i.book_id,
        json_agg(
            json_build_object(
                'user_id', i.user_id,
                'type', i.type,
                'value', i.value
            )
        ) AS interactions
    FROM interaction i
    WHERE i.book_id = ANY(%s)
    GROUP BY i.book_id
"""

# Content-based recommendation
def get_recommended_book_ids(book_id, cos_sim):
    # find the index of the book
    book_index = np.where(cos_sim[:, 0, 0] == book_id)
    if len(book_index[0]) == 0:
        return []
    # get the top 10 similar books
    recommended_books = cos_sim[book_index, 1:11, 0][0][0]
    return list(recommended_books)