"""
Search relevance evaluation and A/B test simulation.

Compares a popularity ranker against item-based collaborative filtering on
MovieTweetings ratings, then simulates search sessions to see whether the
better ranker moves play conversion.

Usage:
    python search_relevance_ab.py data/
"""
import sys
import json
import difflib

import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
TOP_K = 10

# data filtering
RELEVANT_RATING = 8          # MovieTweetings is rated 0-10, 8+ counts as "liked"
N_MOVIES_KEPT = 3000
MIN_RATINGS_PER_USER = 20

# session simulation (these are assumptions, not measured behaviour)
N_SESSIONS = 20000
QUERY_TYPES = ["exact", "genre", "vague", "misspelled"]
QUERY_MIX = [0.45, 0.25, 0.20, 0.10]
P_CLICK_IF_MATCH = 0.85
P_CLICK_IF_MISS = 0.30
P_PLAY_AFTER_CLICK = 0.80
P_TRY_AGAIN = 0.55
P_RETRY_MATCHES = 0.50
MAX_ATTEMPTS = 4

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- data
def load_data(folder):
    ratings = pd.read_csv(folder + "/ratings.dat", sep="::", engine="python",
                          names=["user", "movie", "rating", "time"])
    movies = pd.read_csv(folder + "/movies.dat", sep="::", engine="python",
                         names=["movie", "title", "genres"], encoding="latin-1")
    movies["genres"] = movies["genres"].fillna("Unknown")

    ratings = ratings[ratings["movie"].isin(movies["movie"])]

    # keep the most rated movies and users who rated at least 20 of them
    popular = ratings["movie"].value_counts().head(N_MOVIES_KEPT).index
    ratings = ratings[ratings["movie"].isin(popular)]
    per_user = ratings.groupby("user")["user"].transform("size")
    ratings = ratings[per_user >= MIN_RATINGS_PER_USER].copy()

    # re-number users and movies from 0 so they can index arrays
    ratings["user"] = ratings["user"].astype("category").cat.codes
    movie_index = {m: i for i, m in enumerate(sorted(ratings["movie"].unique()))}
    ratings["movie"] = ratings["movie"].map(movie_index)
    movies = movies[movies["movie"].isin(movie_index)].copy()
    movies["movie"] = movies["movie"].map(movie_index)
    movies = movies.sort_values("movie").reset_index(drop=True)
    return ratings, movies


def split_train_test(ratings):
    """Each user's most recent 20% of ratings become the test set."""
    ratings = ratings.sort_values(["user", "time"])
    ratings["from_end"] = ratings.groupby("user").cumcount(ascending=False)
    n_ratings = ratings.groupby("user")["user"].transform("size")
    n_test = np.maximum(1, (n_ratings * 0.2).astype(int))
    is_test = ratings["from_end"] < n_test
    return ratings[~is_test], ratings[is_test]


# ------------------------------------------------------------- rankers
class Rankers:
    """Two ways to score movies for a user: popularity (control) and item CF (treatment)."""

    def __init__(self, train, n_users, n_movies):
        ones = np.ones(len(train))
        matrix = sparse.csr_matrix((ones, (train["user"], train["movie"])),
                                   shape=(n_users, n_movies))
        self.seen = matrix.toarray() > 0
        self.popularity = np.asarray(matrix.sum(axis=0)).ravel()

        # cosine similarity between movies, based on who watched them
        col_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0))).ravel()
        normed = matrix.multiply(1 / np.maximum(col_norms, 1e-9)).tocsr()
        similarity = (normed.T @ normed).toarray()
        np.fill_diagonal(similarity, 0)
        self.cf_scores = (matrix @ similarity).astype(np.float32)

    def score(self, user, arm):
        """arm 0 = popularity, arm 1 = collaborative filtering. Seen movies are pushed to the bottom."""
        if arm == 0:
            scores = self.popularity.astype(float).copy()
        else:
            scores = self.cf_scores[user].astype(float).copy()
        scores[self.seen[user]] = -np.inf
        return scores


# ------------------------------------------------------ offline metrics
def offline_metrics(rankers, liked, users, arm):
    discounts = 1 / np.log2(np.arange(2, TOP_K + 2))
    precision, ndcg, mrr, hit = [], [], [], []

    for user in users:
        scores = rankers.score(user, arm)
        top = np.argpartition(-scores, TOP_K)[:TOP_K]
        top = top[np.argsort(-scores[top])]
        hits = np.array([m in liked[user] for m in top], dtype=float)

        precision.append(hits.mean())
        hit.append(hits.max())
        best_possible = discounts[:min(TOP_K, len(liked[user]))].sum()
        ndcg.append((hits * discounts).sum() / best_possible)
        mrr.append(1 / (np.argmax(hits) + 1) if hits.any() else 0)

    return {"precision": np.mean(precision), "ndcg": np.mean(ndcg),
            "mrr": np.mean(mrr), "hit_rate": np.mean(hit)}


# ------------------------------------------------- session simulation
def add_typo(title):
    """Drop a character or swap two neighbours somewhere inside the title."""
    pos = rng.integers(1, max(2, len(title) - 2))
    if rng.random() < 0.5:
        return title[:pos] + title[pos + 1:]
    return title[:pos] + title[pos + 1] + title[pos] + title[pos + 2:]


def best_of(scores, candidates, n):
    candidates = np.asarray(candidates)
    order = np.argsort(-scores[candidates])
    return candidates[order[:n]]


def first_search_matches(user, query_type, arm, rankers, liked, titles, genre_lists, movies_by_genre):
    """Does the first results page contain what this user wanted?"""
    target = rng.choice(list(liked[user]))

    if query_type == "exact":
        return True

    if query_type == "misspelled":
        close = difflib.get_close_matches(add_typo(titles[target]), titles, n=1, cutoff=0.0)
        return bool(close) and close[0] == titles[target]

    scores = rankers.score(user, arm)
    if query_type == "genre":
        genre = rng.choice(genre_lists[target])
        shown = best_of(scores, movies_by_genre[genre], 3)
    else:  # vague query, rank everything
        shown = best_of(scores, np.arange(len(titles)), 3)
    return bool(set(shown) & liked[user])


def simulate_sessions(rankers, liked, users, movies):
    titles = movies["title"].tolist()
    genre_lists = movies["genres"].str.split("|").tolist()
    movies_by_genre = {}
    for movie_id, genres in enumerate(genre_lists):
        for g in genres:
            movies_by_genre.setdefault(g, []).append(movie_id)

    # balanced 50/50 split between control and treatment
    arms = rng.permutation(np.arange(N_SESSIONS) % 2)
    rows = []

    for n in range(N_SESSIONS):
        arm = int(arms[n])
        user = rng.choice(users)
        query_type = rng.choice(QUERY_TYPES, p=QUERY_MIX)

        attempts, seconds, played, first_match = 0, 0.0, 0, 0
        for attempt in range(MAX_ATTEMPTS):
            attempts += 1
            seconds += rng.lognormal(2.0, 0.5)

            if attempt == 0:
                matched = first_search_matches(user, query_type, arm, rankers, liked,
                                               titles, genre_lists, movies_by_genre)
                first_match = int(matched)
            else:
                matched = rng.random() < P_RETRY_MATCHES

            p_click = P_CLICK_IF_MATCH if matched else P_CLICK_IF_MISS
            if rng.random() < p_click:
                played = int(rng.random() < P_PLAY_AFTER_CLICK)
                break
            if rng.random() > P_TRY_AGAIN:
                break

        rows.append((arm, query_type, attempts, seconds, played, first_match))

    return pd.DataFrame(rows, columns=["arm", "query_type", "attempts", "seconds",
                                       "played", "first_match"])


# --------------------------------------------------------- A/B test
def analyse_ab_test(sessions):
    control = sessions[sessions["arm"] == 0]
    treatment = sessions[sessions["arm"] == 1]

    # sample ratio check: did the split come out even?
    _, srm_p = stats.chisquare([len(control), len(treatment)])

    n1, n2 = len(control), len(treatment)
    p1, p2 = control["played"].mean(), treatment["played"].mean()
    pooled = (control["played"].sum() + treatment["played"].sum()) / (n1 + n2)
    z = (p2 - p1) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low, ci_high = (p2 - p1) - 1.96 * se, (p2 - p1) + 1.96 * se

    # users needed per arm to detect a 2 point change with 80% power
    z_alpha, z_power = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    n_needed = int(np.ceil(2 * p1 * (1 - p1) * ((z_alpha + z_power) / 0.02) ** 2))

    return {
        "sessions": len(sessions),
        "srm_p_value": srm_p,
        "control_conversion": p1,
        "treatment_conversion": p2,
        "lift_points": 100 * (p2 - p1),
        "z": z,
        "p_value": p_value,
        "ci95_points": [100 * ci_low, 100 * ci_high],
        "n_per_arm_for_2pt_lift": n_needed,
        "avg_attempts": sessions["attempts"].mean(),
        "avg_seconds": sessions["seconds"].mean(),
    }


# -------------------------------------------------------- dashboard
def draw_dashboard(offline, sessions, path="dashboard.png"):
    grey, blue = "#9aa4b2", "#2563eb"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].bar(["Popularity", "Item CF"],
                   [offline["popularity"]["ndcg"], offline["item_cf"]["ndcg"]],
                   color=[grey, blue])
    axes[0, 0].set_title("Offline NDCG@10")

    by_type = (sessions.groupby(["query_type", "arm"])["first_match"].mean()
               .unstack().reindex(QUERY_TYPES))
    by_type.plot.bar(ax=axes[0, 1], color=[grey, blue], legend=False)
    axes[0, 1].set_title("First-search match rate (grey control, blue treatment)")
    axes[0, 1].yaxis.set_major_formatter("{x:.0%}")
    axes[0, 1].tick_params(axis="x", rotation=0)
    axes[0, 1].set_xlabel("")

    funnel = [100, 100 * sessions["played"].mean()]
    axes[1, 0].barh(["Played", "Search opened"], funnel[::-1], color=blue)
    axes[1, 0].set_title("Funnel (% of sessions)")

    attempts = (sessions.groupby("played")["attempts"].value_counts(normalize=True)
                .unstack(0))
    attempts.plot.bar(ax=axes[1, 1], color=[grey, blue], legend=False)
    axes[1, 1].set_title("Searches per session (grey closed, blue played)")
    axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(path, dpi=150)


# ------------------------------------------------------------- main
def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"

    ratings, movies = load_data(folder)
    n_users, n_movies = ratings["user"].nunique(), len(movies)
    print(f"Users: {n_users:,}  Movies: {n_movies:,}  Ratings: {len(ratings):,}")

    train, test = split_train_test(ratings)
    liked_rows = test[test["rating"] >= RELEVANT_RATING]
    liked = liked_rows.groupby("user")["movie"].apply(set).to_dict()
    users = np.array(sorted(liked))

    rankers = Rankers(train, n_users, n_movies)

    offline = {"popularity": offline_metrics(rankers, liked, users, arm=0),
               "item_cf": offline_metrics(rankers, liked, users, arm=1)}
    offline["ndcg_lift_pct"] = 100 * (offline["item_cf"]["ndcg"] / offline["popularity"]["ndcg"] - 1)
    print("\nOffline results at K=10")
    print(json.dumps(offline, indent=2, default=float))

    sessions = simulate_sessions(rankers, liked, users, movies)
    ab = analyse_ab_test(sessions)
    match_rates = (sessions.groupby(["query_type", "arm"])["first_match"].mean()
                   .unstack().round(3))
    print("\nA/B test")
    print(json.dumps(ab, indent=2, default=float))
    print("\nFirst-search match rate by query type (0 = control, 1 = treatment)")
    print(match_rates)

    ab["match_rate_by_query_type"] = match_rates.to_dict()
    with open("results.json", "w") as f:
        json.dump({"offline": offline, "ab_test": ab}, f, indent=2, default=float)

    draw_dashboard(offline, sessions)


if __name__ == "__main__":
    main()
