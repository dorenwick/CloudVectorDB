





class WorksCollectionTrim():
    """
    TODO: We need to generate keywords, from abstract string.

    TODO: every unigram from the authors should be put into a list, unless they are initials.
        And then we can add this as metadata to look through, during our hybrid search system setup.

    TODO: Make an author name list, for both first names and last names, and map them to concepts, and work_id's,
        we do not use initials.

    We build a database in postgresql, and have first names and last names of display names, essentially be indexed
    and have a column for work_ids (a list of work_ids)-we prefer a set, and then also a column for topics and subfields and fields
    associated with each name.



    """
    def __init__(self):
        pass