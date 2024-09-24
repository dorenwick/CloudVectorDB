@measure_time
def generate_hard_negative_triplets(self, batch_size=512, knn=128, min_distance=0.15, max_distance=0.3):
    self.load_index_and_mapping()
    self.load_works_data()
    unigrams_df, bigrams_df = self.load_ngrams()

    common_authors_file = os.path.join(self.datasets_directory, "works_common_authors_filtered.parquet")
    common_authors_df = pl.read_parquet(common_authors_file)

    index_path = self.index_works_file
    index = faiss.read_index(index_path)

    mapping_path = self.id_mapping_works_file
    mapping_df = pl.read_parquet(mapping_path)
    work_id_to_int_id = dict(zip(mapping_df['work_id'], mapping_df['works_int_id']))
    int_id_to_work_id = dict(zip(mapping_df['works_int_id'], mapping_df['work_id']))

    triplets = []

    for batch in common_authors_df.iter_slices(batch_size):
        anchor_ids = batch['work_id_one'].to_list()
        positive_ids = batch['work_id_two'].to_list()

        anchor_int_ids = [work_id_to_int_id[wid] for wid in anchor_ids]
        anchor_strings = [self.create_full_string(self.work_details[wid]) for wid in anchor_ids]
        anchor_embeddings = self.batch_encode_works(anchor_strings)

        distances, indices = self.perform_batch_search(index, anchor_embeddings, knn)

        for i, (anchor_id, positive_id, anchor_embedding) in enumerate(
                zip(anchor_ids, positive_ids, anchor_embeddings)):
            anchor_string = anchor_strings[i]
            positive_string = self.create_full_string(self.work_details[positive_id])
            positive_embedding = self.model.encode(positive_string)

            anchor_positive_distance = np.linalg.norm(anchor_embedding - positive_embedding)

            if anchor_positive_distance < min_distance or anchor_positive_distance > max_distance:
                continue

            candidate_negative_ids = [int_id_to_work_id[int(idx)] for idx in indices[i]]
            candidate_negative_strings = [self.create_full_string(self.work_details[wid]) for wid in
                                          candidate_negative_ids]
            candidate_negative_embeddings = self.batch_encode_works(candidate_negative_strings)

            valid_negatives = []
            for neg_id, neg_string, neg_embedding in zip(candidate_negative_ids, candidate_negative_strings,
                                                         candidate_negative_embeddings):
                anchor_negative_distance = np.linalg.norm(anchor_embedding - neg_embedding)
                positive_negative_distance = np.linalg.norm(positive_embedding - neg_embedding)

                if min_distance <= anchor_negative_distance <= max_distance and \
                        min_distance <= positive_negative_distance <= max_distance:
                    valid_negatives.append((neg_id, neg_string, neg_embedding, anchor_negative_distance))

            if not valid_negatives:
                continue

            valid_negatives.sort(key=lambda x: x[3], reverse=True)  # Sort by distance to anchor, descending
            negative_id, negative_string, negative_embedding, _ = valid_negatives[0]

            anchor_positive_score = self.calculate_total_score(anchor_id, positive_id, anchor_string,
                                                               positive_string)
            anchor_negative_score = self.calculate_total_score(anchor_id, negative_id, anchor_string,
                                                               negative_string)

            if anchor_positive_score <= anchor_negative_score:
                continue

            triplet = {
                'anchor_id': anchor_id,
                'anchor_string': anchor_string,
                'positive_id': positive_id,
                'positive_string': positive_string,
                'negative_id': negative_id,
                'negative_string': negative_string,
                'anchor_positive_distance': float(anchor_positive_distance),
                'anchor_negative_distance': float(np.linalg.norm(anchor_embedding - negative_embedding)),
                'positive_negative_distance': float(np.linalg.norm(positive_embedding - negative_embedding)),
                'anchor_positive_score': float(anchor_positive_score),
                'anchor_negative_score': float(anchor_negative_score),
            }
            triplets.append(triplet)

        if len(triplets) >= self.num_knn_pairs:
            break

    triplets_df = pl.DataFrame(triplets)
    output_file = os.path.join(self.datasets_directory, "hard_negative_triplets.parquet")
    triplets_df.write_parquet(output_file)
    print(f"Generated {len(triplets)} hard negative triplets and saved to {output_file}")


def calculate_total_score(self, work1_id, work2_id, string1, string2):
    common_unigrams = set(string1.lower().split()) & set(string2.lower().split())
    common_bigrams = set(zip(string1.lower().split()[:-1], string1.lower().split()[1:])) & \
                     set(zip(string2.lower().split()[:-1], string2.lower().split()[1:]))

    work1 = self.work_details[work1_id]
    work2 = self.work_details[work2_id]

    common_field = work1['field_string'] == work2['field_string']
    common_subfield = work1['subfield_string'] == work2['subfield_string']

    unigram_score = sum(self.get_gram_scores(common_unigrams, self.unigrams_df).values())
    bigram_score = sum(self.get_gram_scores(common_bigrams, self.bigrams_df).values())

    field_score = 3.0 * common_field
    subfield_score = 1.0 * common_subfield

    return unigram_score + bigram_score + field_score + subfield_score




    @measure_time
    def generate_training_pairs(self, batch_size=512, knn=128, distance_threshold=0.1, min_count=2, max_appearances=8):
        """
        TODO: We will filter out pairs that have far away distances. So for example, we will filter out:
            pairs where the distance threshold for min and max is determined by p-values.

        Generate training pairs using KNN search.

        :param batch_size: Number of works to process in each batch
        :param knn: Number of nearest neighbors to consider
        :param distance_threshold: Maximum distance threshold for similar works
        :return: None
        """


        self.load_index_and_mapping()
        self.load_works_data()
        unigrams_df, bigrams_df = self.load_ngrams()

        works_filtered_df = pl.read_parquet(self.works_all_collected_file)

        pairs_generated = 0
        processed_works = set()

        index_path = self.index_works_file
        index = faiss.read_index(index_path)

        mapping_path = self.id_mapping_works_file
        mapping_df = pl.read_parquet(mapping_path)

        print("Columns in the mapping DataFrame:")
        print(mapping_df.columns)

        faiss_to_works_id = dict(zip(mapping_df['works_int_id'], mapping_df['work_id']))

        cited_by_count_map = dict(zip(works_filtered_df['work_id'], works_filtered_df['cited_by_count']))

        unigrams_dict = dict(zip(self.works_df['work_id'], self.works_df['unigrams']))


        while pairs_generated < (self.num_knn_pairs * 2.0):
            # Then, filter the DataFrame
            unprocessed_work_ids = self.works_df.filter(
                (pl.col('work_id_search_count') == 0) &
                (~pl.col('work_id').is_in(processed_works))
            ).select('work_id').limit(batch_size).to_series().to_list()


            if not unprocessed_work_ids:
                print("No more unprocessed works found.")
                break

            similar_works_df = self.batch_search_similar_works(unprocessed_work_ids, knn, index, faiss_to_works_id,
                                                               distance_threshold=distance_threshold,
                                                               print_distance_stats=True)

            all_pairs = []
            all_distances = []
            work_pair_count = {}
            print("Length of processed works: ", len(processed_works))
            gc.collect()

            for query_work_id in tqdm(unprocessed_work_ids, desc="Processing work IDs"):
                # Use filter instead of boolean indexing
                similar_works = similar_works_df.filter(pl.col('query_work_id') == query_work_id)
                similar_work_ids = similar_works['similar_work_id'].to_list()
                distances = similar_works['distance'].to_list()

                valid_pairs, counts, new_work_pair_count = self.filter_and_count_pairs(similar_work_ids, unigrams_dict,
                                                                                       self.work_details)

                all_pairs.extend(valid_pairs)
                all_distances.extend([distances[similar_work_ids.index(pair[1])] for pair in valid_pairs])

                for work_id, count in new_work_pair_count.items():
                    work_pair_count[work_id] = work_pair_count.get(work_id, 0) + count


            filtered_pairs, filtered_distances = self.filter_pairs_by_count(all_pairs, all_distances, work_pair_count,
                                                                            cited_by_count_map, min_count=min_count)
            print(f"Total pairs after filtering for min_count req of {min_count} or more: {len(filtered_pairs)}")

            all_pairs, all_distances = self.filter_pairs_by_appearance(filtered_pairs, filtered_distances,
                                                                       cited_by_count_map,
                                                                       max_appearances=max_appearances)

            print(f"Total pairs after filtering out max_appearances counts over {max_appearances}: {len(all_pairs)}")

            work_ids = set([work_id for pair in all_pairs for work_id in pair])

            work_details = self.fetch_work_details(work_ids, works_filtered_df)

            vectorized_unigrams, vectorized_bigrams, vectorized_fields, vectorized_subfields = self.process_and_vectorize_common_elements(
                work_details, all_pairs)

            insert_data = self.create_insert_data(all_pairs, work_details, vectorized_unigrams, vectorized_bigrams,
                                                  vectorized_fields, vectorized_subfields, all_distances)
            insert_data = self.calculate_total_scores(insert_data, unigrams_df, bigrams_df)
            insert_data = self.process_p_values(insert_data)

            processed_works.update(unprocessed_work_ids)
            processed_works.update(work_ids)

            self.update_processed_works(unprocessed_work_ids, work_ids)

            self.batch_insert_siamese_data(insert_data)
            self.print_memory_usage("memory usage now after batch_insert_siamese_data")

            pairs_generated += len(insert_data)
            print(f"Generated {pairs_generated} pairs so far. Current knn: {knn}")

            if (pairs_generated >= (self.num_knn_pairs * 2.0)) or len(processed_works) > int(len(self.works_df) * 0.99):
                break

        self.save_processed_data()

        print(f"Total pairs generated: {pairs_generated}")


    @measure_time
    def filter_pairs_by_count(self, all_pairs, all_distances, work_pair_count, cited_by_count_map, min_count=2):
        # Filter pairs and distances based on minimum count
        filtered_pairs_and_distances = [
            (pair, distance) for pair, distance in zip(all_pairs, all_distances)
            if work_pair_count.get(pair[0], 0) >= min_count and work_pair_count.get(pair[1], 0) >= min_count
        ]

        # Sort pairs by combined cited_by_count in descending order
        sorted_pairs_and_distances = sorted(
            filtered_pairs_and_distances,
            key=lambda x: (cited_by_count_map.get(x[0][0], 0) + cited_by_count_map.get(x[0][1], 0)),
            reverse=True
        )

        # Separate the sorted pairs and distances
        sorted_pairs, sorted_distances = zip(*sorted_pairs_and_distances) if sorted_pairs_and_distances else ([], [])

        return list(sorted_pairs), list(sorted_distances)

    @measure_time
    def filter_pairs_by_appearance(self, filtered_pairs, filtered_distances, cited_by_count_map, max_appearances=6):
        final_pairs = []
        final_distances = []
        work_appearance_count = {}

        # Sort pairs and distances by combined cited_by_count in descending order
        sorted_pairs_and_distances = sorted(
            zip(filtered_pairs, filtered_distances),
            key=lambda x: (cited_by_count_map.get(x[0][0], 0) + cited_by_count_map.get(x[0][1], 0)),
            reverse=True
        )

        for pair, distance in sorted_pairs_and_distances:
            if work_appearance_count.get(pair[0], 0) < max_appearances and \
                    work_appearance_count.get(pair[1], 0) < max_appearances:
                final_pairs.append(pair)
                final_distances.append(distance)
                work_appearance_count[pair[0]] = work_appearance_count.get(pair[0], 0) + 1
                work_appearance_count[pair[1]] = work_appearance_count.get(pair[1], 0) + 1

        return final_pairs, final_distances

    @measure_time
    def process_p_values(self, insert_data):
        scores, mean_score, median_score, std_score = self.calculate_score_statistics(insert_data)
        insert_data = self.assign_p_values(insert_data, mean_score, median_score, std_score)
        filtered_data = self.filter_by_p_value(insert_data)

        work_id_count = self.create_work_id_count(filtered_data)
        final_data = self.remove_single_occurrence_pairs(filtered_data, work_id_count)
        self.print_p_value_statistics(final_data)
        return final_data

    @measure_time
    def update_processed_works(self, queried_work_ids, found_work_ids):
        # Combine queried and found work_ids, removing duplicates
        all_work_ids = set(queried_work_ids) | set(found_work_ids)

        # Update work_id_search_count in memory
        for work_id in all_work_ids:
            self.work_id_search_count[work_id] = self.work_id_search_count.get(work_id, 0) + 1

        # Update work_id_search_count in the DataFrame using polars syntax
        self.works_df = self.works_df.with_columns([
            pl.when(pl.col('work_id').is_in(all_work_ids))
            .then(pl.col('work_id_search_count') + 1)
            .otherwise(pl.col('work_id_search_count'))
            .alias('work_id_search_count')
        ])

        print(f"Updated work_id_search_count for {len(all_work_ids)} works")

    @measure_time
    def calculate_total_scores(self, insert_data, unigrams_df, bigrams_df):
        """
        Calculate total scores using Polars, with modifications as per TODO comments.
        We need to make test vectorizated gram scores because loading up the dictionary in this method takes a long time.

        """
        # Convert insert_data to a Polars DataFrame if it's not already
        if not isinstance(insert_data, pl.DataFrame):
            df = pl.DataFrame(insert_data)
        else:
            df = insert_data

        # Calculate gram scores
        df = df.with_columns([
            self.vectorized_gram_scores('common_unigrams', unigrams_df, testing_method=True).alias('unigram_score'),
            self.vectorized_gram_scores('common_bigrams', bigrams_df, testing_method=True).alias('bigram_score')
        ])

        # Calculate sum of gram scores instead of average
        df = df.with_columns([
            (pl.col('unigram_score') + pl.col('bigram_score')).alias('sum_gram_score')
        ])

        scalar_multiplier = 0.05
        df = df.with_columns([
            (pl.when(pl.col('common_field') >= 0)
             .then(pl.col('common_field') * (3.0 + 2.0 * scalar_multiplier * pl.col('sum_gram_score')))
             .otherwise(0)).alias('field_score'),
            (pl.when(pl.col('common_subfield') >= 0)
             .then(pl.col('common_subfield') * (1.0 + scalar_multiplier * pl.col('sum_gram_score')))
             .otherwise(0)).alias('subfield_score')
        ])

        # Calculate total score
        df = df.with_columns([
            (pl.col('unigram_score') + pl.col('bigram_score') + pl.col('field_score') + pl.col('subfield_score')).alias(
                'total_score')
        ])

        # Convert back to list of dictionaries
        return df.to_dicts()

    @measure_time
    def vectorized_gram_scores(self, gram_column, gram_df, testing_method=False):
        """
        Calculate vectorized gram scores using Polars.
        If testing_method is True, return random scores instead of actual calculations.


        """
        if testing_method:
            # Define a function to return a random score between 0 and 5
            def calculate_random_score(gram_list):
                return random.uniform(0, 5) * len(gram_list)

            # Use pl.col().map() to apply the random score function to each list in the column
            return pl.col(gram_column).map_elements(lambda x: calculate_random_score(x))
        else:
            gram_type = "unigram_type" if 'unigram_type' in gram_df.columns else "bigram_type"

            # Create a dictionary of gram scores
            scores_dict = dict(zip(gram_df[gram_type], gram_df['score']))

            # Define a function to calculate the score for a list of grams
            def calculate_score(gram_list):
                return sum(scores_dict.get(gram, 2.5) for gram in gram_list)

            # Use pl.col().map() to apply the function to each list in the column
            return pl.col(gram_column).map_elements(lambda x: calculate_score(x))


    @measure_time
    def calculate_score_statistics(self, insert_data):
        scores = [item['total_score'] for item in insert_data]
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        return scores, mean_score, median_score, std_score

    @measure_time
    def assign_p_values(self, insert_data, mean_score, median_score, std_score):
        # Convert insert_data to a DataFrame if it's not already
        if not isinstance(insert_data, pl.DataFrame):
            df = pl.DataFrame(insert_data)
        else:
            df = insert_data.clone()

        # Vectorized calculation of z_scores and p_values
        df = df.with_columns([
            ((pl.col('total_score') - mean_score) / std_score).alias('z_score')
        ])

        df = df.with_columns([
            (1 - pl.col('z_score').map_elements(norm.cdf)).alias('p_value')
        ])

        # If the original input was a list of dictionaries, convert back
        if not isinstance(insert_data, pl.DataFrame):
            return df.to_dicts()
        else:
            return df

    @measure_time
    def filter_by_p_value(self, insert_data):
        if isinstance(insert_data, pl.DataFrame):
            return insert_data.filter((pl.col('p_value') <= 0.49) | (pl.col('p_value') >= 0.51))
        elif isinstance(insert_data, list):
            df = pl.DataFrame(insert_data)
            filtered_df = df.filter((pl.col('p_value') <= 0.49) | (pl.col('p_value') >= 0.51))
            return filtered_df.to_dicts()
        else:
            raise TypeError("insert_data must be a DataFrame or a list of dictionaries")

    @measure_time
    def remove_single_occurrence_pairs(self, filtered_data, work_id_count):
        if not isinstance(filtered_data, pl.DataFrame):
            df = pl.DataFrame(filtered_data)
        else:
            df = filtered_data

        def has_both_occurrences(work_id):
            return work_id_count[work_id]['above'] > 0 and work_id_count[work_id]['below'] > 0

        filtered_df = df.filter(
            pl.col('work_id_one').map_elements(has_both_occurrences) &
            pl.col('work_id_two').map_elements(has_both_occurrences)
        )

        if not isinstance(filtered_data, pl.DataFrame):
            return filtered_df.to_dicts()
        else:
            return filtered_df

    @measure_time
    def create_work_id_count(self, filtered_data):
        if not isinstance(filtered_data, pl.DataFrame):
            df = pl.DataFrame(filtered_data)
        else:
            df = filtered_data

        work_id_count = {}

        for row in df.iter_rows(named=True):
            for work_id in [row['work_id_one'], row['work_id_two']]:
                if work_id not in work_id_count:
                    work_id_count[work_id] = {'above': 0, 'below': 0}
                if row['p_value'] > 0.5:
                    work_id_count[work_id]['above'] += 1
                else:
                    work_id_count[work_id]['below'] += 1

        return work_id_count

    def filter_and_count_pairs(self, similar_works, unigrams_dict, work_details):
        """
        Filter and count pairs of works based on common unigrams and fields.

        :param similar_works: List of similar work IDs
        :param unigrams_dict: Dictionary of work IDs to unigrams
        :param work_details: Dictionary of work details
        :return: Tuple of valid pairs, counts, and work pair count
        """
        common_stop_words = self.get_stop_words()
        possible_pairs = list(combinations(similar_works, 2))
        random_numbers = np.random.random(len(possible_pairs))
        valid_pairs = []
        counts = {"common_3": 0, "common_2": 0, "common_1": 0, "common_field_subfield": 0}
        work_pair_count = {}
        pair_conditions = {}

        for idx, (work1_id, work2_id) in enumerate(possible_pairs):
            work1 = work_details.get(work1_id, {})
            work2 = work_details.get(work2_id, {})
            work1_unigrams = set(unigrams_dict.get(work1_id, [])) - common_stop_words
            work2_unigrams = set(unigrams_dict.get(work2_id, [])) - common_stop_words
            common_unigrams_count = len(work1_unigrams & work2_unigrams)
            common_field = work1.get('field_string') == work2.get('field_string')
            rand_num = random_numbers[idx]

            pair_key = tuple(sorted([work1_id, work2_id]))
            condition = None

            if common_unigrams_count >= 3:
                condition = "common_3"
                counts["common_3"] += 1
            elif common_unigrams_count >= 2 and rand_num > 0.05:
                condition = "common_2"
                counts["common_2"] += 1
            elif common_unigrams_count >= 1 and rand_num > 0.90:
                condition = "common_1"
                counts["common_1"] += 1
            elif common_field and rand_num > 0.90:
                condition = "common_field_subfield"
                counts["common_field_subfield"] += 1
            elif rand_num > 0.999:
                condition = "random"

            if condition:
                if pair_key not in pair_conditions or condition in ["common_3", "common_2"]:
                    pair_conditions[pair_key] = condition
                    valid_pairs.append((work1_id, work2_id))
                    work_pair_count[work1_id] = work_pair_count.get(work1_id, 0) + 1
                    work_pair_count[work2_id] = work_pair_count.get(work2_id, 0) + 1

        return valid_pairs, counts, work_pair_count

    def get_stop_words(self):
        """get a bigger list of stop words for us here"""

        # You can expand this set of stop words as needed
        return {
            'a', 'A', 'about', 'About', 'above', 'Above', 'after', 'After', 'again', 'Again', 'against', 'Against',
            'all', 'All', 'am', 'Am', 'an', 'An', 'and', 'And', 'any', 'Any', 'are', 'Are', 'as', 'As', 'at', 'At',
            'be', 'Be', 'because', 'Because', 'been', 'Been', 'before', 'Before', 'being', 'Being', 'below', 'Below',
            'between', 'Between', 'both', 'Both', 'but', 'But', 'by', 'By', 'can', 'Can', 'did', 'Did', 'do', 'Do',
            'does', 'Does', 'doing', 'Doing', 'down', 'Down', 'during', 'During', 'each', 'Each', 'few', 'Few',
            'for', 'For', 'from', 'From', 'further', 'Further', 'had', 'Had', 'has', 'Has', 'have', 'Have',
            'having', 'Having', 'he', 'He', 'her', 'Her', 'here', 'Here', 'hers', 'Hers', 'herself', 'Herself',
            'him', 'Him', 'himself', 'Himself', 'his', 'His', 'how', 'How', 'i', 'I', 'if', 'If', 'in', 'In',
            'into', 'Into', 'is', 'Is', 'it', 'It', 'its', 'Its', 'itself', 'Itself', 'just', 'Just', 'me', 'Me',
            'more', 'More', 'most', 'Most', 'my', 'My', 'myself', 'Myself', 'no', 'No', 'nor', 'Nor', 'not', 'Not',
            'now', 'Now', 'of', 'Of', 'off', 'Off', 'on', 'On', 'once', 'Once', 'only', 'Only', 'or', 'Or',
            'other', 'Other', 'our', 'Our', 'ours', 'Ours', 'ourselves', 'Ourselves', 'out', 'Out', 'over', 'Over',
            'own', 'Own', 'same', 'Same', 'she', 'She', 'should', 'Should', 'so', 'So', 'some', 'Some', 'such', 'Such',
            'than', 'Than', 'that', 'That', 'the', 'The', 'their', 'Their', 'theirs', 'Theirs', 'them', 'Them',
            'themselves', 'Themselves', 'then', 'Then', 'there', 'There', 'these', 'These', 'they', 'They',
            'this', 'This', 'those', 'Those', 'through', 'Through', 'to', 'To', 'too', 'Too', 'under', 'Under',
            'until', 'Until', 'up', 'Up', 'very', 'Very', 'was', 'Was', 'we', 'We', 'were', 'Were', 'what', 'What',
            'when', 'When', 'where', 'Where', 'which', 'Which', 'while', 'While', 'who', 'Who', 'whom', 'Whom',
            'why', 'Why', 'with', 'With', 'would', 'Would', 'you', 'You', 'your', 'Your', 'yours', 'Yours',
            'yourself', 'Yourself', 'yourselves', 'Yourselves', ',', '.', ':', ';', '!', '?', '"',
            "'", '(', ')', '[', ']', '{', '}', '-', '–', '—', '/', '|', '@', '#',
            '$', '%', '^', '&', '*', '+', '=', '<', '>', '`', '~'
        }


    def load_ngrams(self):
        unigrams_df = pl.read_parquet(self.unigram_data_file)
        bigrams_df = pl.read_parquet(self.bigram_data_file)
        return unigrams_df, bigrams_df