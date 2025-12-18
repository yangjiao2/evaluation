import random

from collections import Counter

from math import sqrt

import evaluate

def rouge_score(reference, prediction):
    # Load the ROUGE evaluation metric
    rouge = evaluate.load('rouge')


    # Compute the ROUGE score
    results = rouge.compute(predictions=[prediction], references=[reference])
    print("rouge score results", results)
    return results

#
#
# def execute_in_batch():
#     batch = []
#     batch.extend([agent.acall(example["inputs"]) for agent in agents])
#     if len(batch) >= concurrency_level:
#         batch_results = await asyncio.gather(*batch, return_exceptions=True)
#         results.extend(list(zip(*[iter(batch_results)] * 2)))
#         batch = []
#
#
# def predict_preferences(dataset, results) -> list:
#     preferences = []
#
#     for example, (res_a, res_b) in zip(dataset, results):
#         input_ = example["inputs"]
#         # Flip a coin to reduce persistent position bias
#         if random.random() < 0.5:
#             pred_a, pred_b = res_a, res_b
#             a, b = "a", "b"
#         else:
#             pred_a, pred_b = res_b, res_a
#             a, b = "b", "a"
#         eval_res = eval_chain.evaluate_string_pairs(
#             prediction=pred_a["output"] if isinstance(pred_a, dict) else str(pred_a),
#             prediction_b=pred_b["output"] if isinstance(pred_b, dict) else str(pred_b),
#             input=input_,
#         )
#         if eval_res["value"] == "A":
#             preferences.append(a)
#         elif eval_res["value"] == "B":
#             preferences.append(b)
#         else:
#             preferences.append(None)  # No preference
#     return preferences
#
#
#
#
#
# def wilson_score_interval(
#     preferences: list, which: str = "a", z: float = 1.96
# ) -> tuple:
#     """Estimate the confidence interval using the Wilson score.
#
#     See: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
#     for more details, including when to use it and when it should not be used.
#     """
#     total_preferences = preferences.count("a") + preferences.count("b")
#     n_s = preferences.count(which)
#
#     if total_preferences == 0:
#         return (0, 0)
#
#     p_hat = n_s / total_preferences
#
#     denominator = 1 + (z**2) / total_preferences
#     adjustment = (z / denominator) * sqrt(
#         p_hat * (1 - p_hat) / total_preferences
#         + (z**2) / (4 * total_preferences * total_preferences)
#     )
#     center = (p_hat + (z**2) / (2 * total_preferences)) / denominator
#     lower_bound = min(max(center - adjustment, 0.0), 1.0)
#     upper_bound = min(max(center + adjustment, 0.0), 1.0)
#
#     return (lower_bound, upper_bound)
#
# from langsmith.schemas import Example, Run
# from langsmith.evaluation import EvaluationResult, run_evaluator
#
# def exact_match(run: Run, example: Example):
#     # "output" is the key we assigned in the create_examples step above
#     expected = example.outputs["output"]
#     predicted = run.outputs["output"]
#     return {"key": "exact_match", "score": predicted == expected}
#
#
# # The name of the test you have already run.
# # This is DISTINCT from the dataset name
# # compute_test_metrics(test_name, evaluators=[exact_match])
#
#
# #
# # preferences = predict_preferences(dataset, results)
# # counts = Counter(preferences)
# # pref_ratios = {k: v / len(preferences) for k, v in counts.items()}
# # for k, v in pref_ratios.items():
# #     print(f"{name_map.get(k)}: {v:.2%}")
