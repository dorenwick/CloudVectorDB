




class KeyWordDatasetConstructor():
    """
    This will construct the entire dataset of keywords so we can extract them from title or abstract with O(1)
    process cpu time and lookup on a Binary Tree.

    We shall want, every single keyword associated with topics from openalex_concepts_topics
    Every single level 2,3,4,5 concept from openalex_concepts_topics
    The filtered unigrams, bigrams, trigrams, and 4+grams from postgresql table.

    high ctf_idf scoring bigrams from the full_bigrams table, where the count is greater than 10.

    also some processed

    """


    def __init__(self):
        pass