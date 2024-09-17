




class KeyWordDatasetConstructor():
    """
    This will construct the entire dataset of keywords so we can extract them from title or abstract with O(1)
    process cpu time and lookup on a Binary Tree.

    We shall want, every single keyword associated with topics from openalex_concepts_topics
    Every single level 2,3,4,5 concept from openalex_concepts_topics
    The filtered unigrams, bigrams, trigrams, and 4+grams from postgresql table.

    high ctf_idf scoring bigrams from the full_bigrams table, where the count is greater than 10.

    also some processed

    TODO: Once we have assembled the parquet file for keyphrases (including unigrams, bigrams, trigrams, and fourgrams)
        we will then need to get the processed batch data and fill in the details of
            batch['keywords_title'] = [[] for _ in range(len(batch))]
            batch['keywords_abstract'] = [[] for _ in range(len(batch))]

    We will run this on the cloud using vectorization and multiprocessing.





    """


    def __init__(self):
        pass