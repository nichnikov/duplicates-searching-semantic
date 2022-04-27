from embeddings import Embedding, transformer_func
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def duplicates_search_func(searched_ids, searched_texts, ids, texts, min_score):
    """Function for searching duplicates in two texts collections with groups numbers.
    searched_ids - IDs of texts to be found
    searched_texts - texts to be found
    ids - IDs of texts in which we will search
    texts - texts in which we will search
    min_score - similarity coefficient
    """

    transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embedder = Embedding(transformer_model, transformer_func)

    """Vectorization"""
    texts_vectors = embedder(texts)
    searched_texts_vectors = embedder(searched_texts)

    """Duplicates Searching"""
    distances_arr = cosine_similarity(texts_vectors, searched_texts_vectors, dense_output=False).T

    search_results = []
    # Duplicates = namedtuple("Duplicate", "searched_text, searched_id, similar_text, similar_text_id, score")
    searched_texts_ids = zip(searched_texts, searched_ids)
    for srch_tx_ids, distances in zip(searched_texts_ids, distances_arr):
        initial_texts_ids = zip(texts, ids)
        search_results += [(srch_tx_ids[0], srch_tx_ids[1], tx_ids[0], tx_ids[1], score) for
                           tx_ids, score in zip(initial_texts_ids, distances) if score >= min_score]

    return sorted(search_results, key=lambda x: x[4], reverse=True)
