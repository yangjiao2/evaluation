import numpy as np
import nltk
import asyncio
from scipy import stats
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")


# Example token-counting functions for different models (replace with actual implementations)
def num_tokens_from_string(text, model_name=None):
    return len(str(text).split())  # Example simple token count


def calculate_significance(statistics_current, statistics_reference):
    """
    Perform a paired t-test to see if there's a statistically significant difference between
    statistics_current and statistics_reference.

    @param statistics_current: A list or series of numbers summarizing the responses from df_current
    @param statistics_reference: A list or series of numbers summarizing the responses from df_reference
    """
    # Two groups of ratings from the same subjects
    try:
        group1_ratings = np.array(statistics_current)
        group2_ratings = np.array(statistics_reference)

        # Perform paired t-test (within-subject t-test)
        t_statistic, p_value = stats.ttest_rel(group1_ratings, group2_ratings, nan_policy="omit")

        # Print the results
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)

        # Check if the difference is statistically significant
        alpha = 0.05
        if p_value < alpha:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error performing paired t-test: {e}")


def calculate_verbosity_stats(answer_column, model):
    """
    Calculate basic verbosity/readability stats for a set of responses.

    @param answer_column: a list/series of responses for which we want to calculate statistics
    """
    word_counts = []
    paragraph_counts = []
    sentence_num = []
    char_length = []
    in_text_citations = []
    char_sentence = []
    word_sentence = []
    char_paragraph = []
    word_paragraph = []
    tokens = []


    for answer in answer_column:
        try:
            answer = str(answer)
            print ("verbosity answer:", answer)
            # Count words
            word_count = len(answer.split())
            word_counts.append(word_count)

            # Count paragraphs
            paragraphs = answer.strip().split("\n\n")
            paragraph_count = len(paragraphs)
            paragraph_counts.append(paragraph_count)

            # Count sentences
            sentences = sent_tokenize(answer)
            # print ("verbosity sentences", sentences)
            sentence_count = len(sentences)
            sentence_num.append(sentence_count)

            # Count characters
            char_count = len(answer)
            char_length.append(char_count)

            # Count in-text citations
            citations = [word for word in answer.split() if "http" in word]
            in_text_citations.append(len(citations))

            # Average characters per sentence
            avg_chars_per_sentence = np.mean([len(sentence) for sentence in sentences]) if sentences else 0
            char_sentence.append(avg_chars_per_sentence)

            # Average words per sentence
            avg_words_per_sentence = np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0
            word_sentence.append(avg_words_per_sentence)

            # Average characters per paragraph
            avg_chars_per_paragraph = np.mean([len(paragraph) for paragraph in paragraphs]) if paragraphs else 0
            char_paragraph.append(avg_chars_per_paragraph)

            # Average words per paragraph
            avg_words_per_paragraph = np.mean([len(paragraph.split()) for paragraph in paragraphs]) if paragraphs else 0
            word_paragraph.append(avg_words_per_paragraph)

            # Token count using the provided synchronous token-counting function
            num_tokens = len(word_tokenize(answer))
            tokens.append(num_tokens)


        except Exception as e:
            print(f"Error calculating verbosity/readability stats: {e}")
    return (
        word_counts,
        paragraph_counts,
        sentence_num,
        char_length,
        in_text_citations,
        char_sentence,
        word_sentence,
        char_paragraph,
        word_paragraph,
        tokens,
    )


def get_verbosity_statistics_single(answer_column, df, model):
    """
    Calculate verbosity statistics for a list or series of responses.

    @param answer_column: a list/series of responses for which we want to calculate statistics
    """
    try:
        (
            word_counts,
            paragraph_counts,
            sentence_num,
            char_length,
            in_text_citations,
            char_sentence,
            word_sentence,
            char_paragraph,
            word_paragraph,
            tokens,
        ) = calculate_verbosity_stats(answer_column=answer_column, model=model)

        return {
            "Words": word_counts,
            "Paragraphs": paragraph_counts,
            "Sentences": sentence_num,
            "Character Length": char_length,
            "In-text Citations": in_text_citations,
            "Characters per Sentence": char_sentence,
            "Words per Sentence": word_sentence,
            "Characters per Paragraph": char_paragraph,
            "Words per Paragraph": word_paragraph,
            "Tokens": tokens,
        }


    except Exception as err:
        print(f"Encountered an  error calculating verbosity statistics: {err}")


def get_verbosity_statistics_comparison(answer_column_curr, answer_column_ref, model):
    """
    Calculate verbosity statistics for a list or series of responses.

    @param answer_column: a list/series of responses for which we want to calculate statistics
    """
    try:
        (
            word_counts_curr,
            paragraph_counts_curr,
            sentence_num_curr,
            char_length_curr,
            in_text_citations_curr,
            char_sentence_curr,
            word_sentence_curr,
            char_paragraph_curr,
            word_paragraph_curr,
            tokens_curr,
        ) = calculate_verbosity_stats(answer_column_curr, model=model)
        (
            word_counts_ref,
            paragraph_counts_ref,
            sentence_num_ref,
            char_length_ref,
            in_text_citations_ref,
            char_sentence_ref,
            word_sentence_ref,
            char_paragraph_ref,
            word_paragraph_ref,
            tokens_ref,
        ) = calculate_verbosity_stats(answer_column_ref, model=model)

        t_value_label = "difference(t-value)"
        p_value_label = "probability(p-value)"

        def get_p_significance_value(curr, ref, t_label, p_label):
            group1_ratings = np.array(curr)
            group2_ratings = np.array(ref)

            # Perform paired t-test (within-subject t-test)
            t_stat, p_value = stats.ttest_rel(group1_ratings, group2_ratings, nan_policy="omit")
            return {
                t_label: round(t_stat, 2),
                p_label:  round(p_value, 2),
            }

        return {
            "Words count":
                get_p_significance_value(word_counts_curr, word_counts_ref, \
                                         t_value_label, p_value_label),
            "Paragraphs":
                get_p_significance_value(paragraph_counts_curr, paragraph_counts_ref, \
                                         t_value_label, p_value_label),
            "Sentences": get_p_significance_value(sentence_num_curr, sentence_num_ref, \
                                                  t_value_label, p_value_label),
            "Character Length": get_p_significance_value(char_length_curr, char_length_ref, \
                                                         t_value_label, p_value_label),
            "In-text Citations": get_p_significance_value(in_text_citations_curr, in_text_citations_ref, \
                                                          t_value_label, p_value_label),
            "Characters per Sentence":
                get_p_significance_value(char_sentence_curr, char_sentence_ref, \
                                         t_value_label, p_value_label),
            "Words per Sentence":
                get_p_significance_value(word_sentence_curr, word_sentence_ref, \
                                         t_value_label, p_value_label),
            "Characters per Paragraph":
                get_p_significance_value(char_paragraph_curr, char_paragraph_ref, \
                                         t_value_label, p_value_label),
            "Words per Paragraph":
                get_p_significance_value(word_paragraph_curr, word_paragraph_ref, \
                                         t_value_label, p_value_label),
            "Tokens":
                get_p_significance_value(tokens_curr, tokens_ref, \
                                         t_value_label, p_value_label),
        }

    except Exception as err:
        print(f"Encountered an  error calculating verbosity statistics: {err}")
