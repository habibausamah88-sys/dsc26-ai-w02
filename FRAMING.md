
## Decision

Rawaj Pay should use risk scores to prioritize transfers for human review, not treat a score as proof of fraud. The threshold should reflect the review team's capacity and the relative cost of false positives and false negatives.

## Metric

Precision measures how many flagged transfers are positive. Recall measures how many positive transfers are detected. Precision@K is useful when analysts can review only the K highest-scoring transfers. Expected cost makes the trade-off explicit by weighting false positives and false negatives.

## Baseline

Start by auditing missing values, duplicate records, malformed timestamps, constant columns, and possible target leakage. Compare a simple transparent baseline with a classical classifier using held-out data. Evaluate precision, recall, Precision@K, and expected cost.

## Classical ML or LLM

A classical tabular model is the more suitable starting point for structured transaction data with labelled outcomes. An LLM might help summarize investigation context, but it should not be the default scoring engine without evidence of improvement. Human review, privacy safeguards, and monitoring remain important.
