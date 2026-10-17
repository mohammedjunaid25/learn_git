"""Part C: retrieve the most relevant policy using TF-IDF cosine similarity."""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_PATH = Path("fraud_policies.json")
QUERY_PATH = Path("retrieval_queries.json")
MIN_SCORE = 0.16


def _load():
    policies = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    docs = [f"{p['title']} {p['text']}" for p in policies]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    return policies, vectorizer, matrix


def retrieve(query):
    """Return {policy_id, title, text, score} or None when relevance is too low."""
    if not isinstance(query, str) or not query.strip():
        return None
    policies, vectorizer, matrix = _load()
    query_vec = vectorizer.transform([query.strip()])
    scores = cosine_similarity(query_vec, matrix).ravel()
    best = int(scores.argmax())
    score = float(scores[best])
    if score < MIN_SCORE:
        return None
    return {**policies[best], "score": score}


def main():
    queries = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    print("PART C POLICY RETRIEVAL")
    print("query | expected | retrieved | score | match")
    correct = 0
    for row in queries:
        result = retrieve(row["query"])
        got = result["policy_id"] if result else "NONE"
        expected = row["expected_policy"] or "NONE"
        match = got == expected
        correct += int(match)
        score_text = f"{result['score']:.3f}" if result else "0.000"
        print(f"{row['query']} | {expected} | {got} | {score_text} | {match}")
    print(f"retrieval_matches={correct}/{len(queries)}")
    print(f"min_score={MIN_SCORE:.2f}")


if __name__ == "__main__":
    main()
