





class KeyWordBuilder():

    """

    Here we shall describe how build up the keywords and their scoring system.

    1 Firstly, we will have to construct and initial potential keywords dataframe and parquet file,
    which should contain around 1 million to 5 million keywords.

    We shall get them from various sources, as described in KeyWordDatasetConstructor.

    2 We will have to go through all the CloudDatasetConstructionEncoder to build
    all the work files, with abstracts and everything, on an initial run locally. We will have to build all the abstract files.

    We then go through all the abstract_string
    and process all the keywords for each abstract file.
    do this by multiprocessing batches together, in polars.
    so we do many batches at the same time. Figure out how to run this efficiently.

    For each keyword we process, we will fill in a count to a refined unigrams, bigrams, trigrams, and four-grams file.
    Where we will have a vector that counts each subfield and each topic. Make sure the ID system is consistent with previous ones used.

    Once this is done, and we have new keyword files, we shall create the scorings for them based off tf-icf, and then perhaps smooth their scoring
    a bit.

    It is only after this step that we will be able to score keywords in the total_score calculate method
    for

    display_name + authors + field + subfield + topic + keywords






    """

    def __init__(self):
        pass