from .database import config_database

def get_interactions_by_userid(user_id):
    interactions_query = """
    SELECT
        i.book_id
    FROM interaction i
    WHERE i.user_id = %s
    """
    result = config_database.query_db(interactions_query, (user_id,))
    book_ids = [row['book_id'] for row in result]
    print(book_ids,'rated')
    return book_ids

def get_books_by_ids(recommend_books):
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

    # Truy vấn thông tin sách và tác giả từ cơ sở dữ liệu
    books = config_database.query_db(books_query, (recommend_books,))

    # Truy vấn thông tin nguồn từ cơ sở dữ liệu
    source_ids = [book["source_id"] for book in books]

    sources = {
        source["id"]: source for source in config_database.query_db(sources_query, (source_ids,))
    }

    # Truy vấn thông tin tương tác từ cơ sở dữ liệu
    interactions = {
        interaction["book_id"]: interaction["interactions"]
        for interaction in config_database.query_db(interactions_query, (recommend_books,))
    }

    # Kết hợp thông tin nguồn và interactions vào dữ liệu sách
    for book in books:
        book["source"] = sources.get(
            book["source_id"], {"id": book["source_id"], "name": "Unknown"}
        )
        book["interactions"] = interactions.get(book["id"], [])

    return books