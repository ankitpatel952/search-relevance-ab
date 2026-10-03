# Popularity vs item-based collaborative filtering on MovieTweetings,
# then a simulated A/B test on search sessions.
#
#   python search_relevance_ab.py data/

import sys
import json
import difflib

import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

K = 10
LIKED = 8            # ratings are 0-10, so 8+ means they liked it
N_SESSIONS = 20000

# made-up user behaviour for the simulation. change these and rerun
QUERY_TYPES = ["exact", "genre", "vague", "misspelled"]
QUERY_MIX = [0.45, 0.25, 0.20, 0.10]
P_CLICK_HIT = 0.85
P_CLICK_MISS = 0.30
P_PLAY = 0.80
P_RETRY = 0.55
P_RETRY_HIT = 0.50
MAX_TRIES = 4

folder = sys.argv[1] if len(sys.argv) > 1 else "data"

ratings = pd.read_csv(folder + "/ratings.dat", sep="::", engine="python",
                      names=["user", "movie", "rating", "time"])
movies = pd.read_csv(folder + "/movies.dat", sep="::", engine="python",
                     names=["movie", "title", "genres"], encoding="latin-1")
movies["genres"] = movies["genres"].fillna("Unknown")
ratings = ratings[ratings["movie"].isin(movies["movie"])]

# only the 3000 most rated movies, and users with at least 20 ratings
top_movies = ratings["movie"].value_counts().head(3000).index
ratings = ratings[ratings["movie"].isin(top_movies)]
n_per_user = ratings.groupby("user")["user"].transform("size")
ratings = ratings[n_per_user >= 20].copy()

# renumber from 0 so the ids can be used as array positions
ratings["user"] = ratings["user"].astype("category").cat.codes
movie_ids = {m: i for i, m in enumerate(sorted(ratings["movie"].unique()))}
ratings["movie"] = ratings["movie"].map(movie_ids)
movies = movies[movies["movie"].isin(movie_ids)].copy()
movies["movie"] = movies["movie"].map(movie_ids)
movies = movies.sort_values("movie").reset_index(drop=True)

n_users = ratings["user"].nunique()
n_movies = len(movies)
print(f"Users: {n_users:,}  Movies: {n_movies:,}  Ratings: {len(ratings):,}")

# last 20% of each user's ratings (by time) go to the test set
ratings = ratings.sort_values(["user", "time"])
ratings["from_end"] = ratings.groupby("user").cumcount(ascending=False)
n_each = ratings.groupby("user")["user"].transform("size")
in_test = ratings["from_end"] < np.maximum(1, (n_each * 0.2).astype(int))
train, test = ratings[~in_test], ratings[in_test]

liked = test[test["rating"] >= LIKED].groupby("user")["movie"].apply(set).to_dict()
users = np.array(sorted(liked))

# the two rankers
X = sparse.csr_matrix((np.ones(len(train)), (train["user"], train["movie"])),
                      shape=(n_users, n_movies))
seen = X.toarray() > 0
popularity = np.asarray(X.sum(axis=0)).ravel()

# item-item cosine similarity from who watched what
norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0))).ravel()
Xn = X.multiply(1 / np.maximum(norms, 1e-9)).tocsr()
sim = (Xn.T @ Xn).toarray()
np.fill_diagonal(sim, 0)
cf = (X @ sim).astype(np.float32)


def scores_for(user, arm):
    # arm 0 = popularity, arm 1 = CF. movies they already saw sink to the bottom
    s = popularity.astype(float).copy() if arm == 0 else cf[user].astype(float).copy()
    s[seen[user]] = -np.inf
    return s


def offline_metrics(arm):
    discount = 1 / np.log2(np.arange(2, K + 2))
    prec, ndcg, mrr, hit = [], [], [], []
    for u in users:
        s = scores_for(u, arm)
        top = np.argpartition(-s, K)[:K]
        top = top[np.argsort(-s[top])]
        hits = np.array([m in liked[u] for m in top], dtype=float)
        prec.append(hits.mean())
        hit.append(hits.max())
        ideal = discount[:min(K, len(liked[u]))].sum()
        ndcg.append((hits * discount).sum() / ideal)
        mrr.append(1 / (np.argmax(hits) + 1) if hits.any() else 0)
    return {"precision": np.mean(prec), "ndcg": np.mean(ndcg),
            "mrr": np.mean(mrr), "hit_rate": np.mean(hit)}


offline = {"popularity": offline_metrics(0), "item_cf": offline_metrics(1)}
offline["ndcg_lift_pct"] = 100 * (offline["item_cf"]["ndcg"] / offline["popularity"]["ndcg"] - 1)
print("\nOffline results at K=10")
print(json.dumps(offline, indent=2, default=float))

# search session simulation
titles = movies["title"].tolist()
genre_lists = movies["genres"].str.split("|").tolist()
by_genre = {}
for i, gl in enumerate(genre_lists):
    for g in gl:
        by_genre.setdefault(g, []).append(i)


def typo(title):
    pos = rng.integers(1, max(2, len(title) - 2))
    if rng.random() < 0.5:
        return title[:pos] + title[pos + 1:]                              # drop a letter
    return title[:pos] + title[pos + 1] + title[pos] + title[pos + 2:]    # swap two


def top_n(s, candidates, n):
    candidates = np.asarray(candidates)
    return candidates[np.argsort(-s[candidates])[:n]]


def first_page_has_it(user, qtype, arm):
    target = rng.choice(list(liked[user]))
    if qtype == "exact":
        return True
    if qtype == "misspelled":
        close = difflib.get_close_matches(typo(titles[target]), titles, n=1, cutoff=0.0)
        return bool(close) and close[0] == titles[target]
    s = scores_for(user, arm)
    if qtype == "genre":
        shown = top_n(s, by_genre[rng.choice(genre_lists[target])], 3)
    else:
        shown = top_n(s, np.arange(n_movies), 3)
    return bool(set(shown) & liked[user])


arms = rng.permutation(np.arange(N_SESSIONS) % 2)    # even split
rows = []
for n in range(N_SESSIONS):
    arm = int(arms[n])
    user = rng.choice(users)
    qtype = rng.choice(QUERY_TYPES, p=QUERY_MIX)

    tries, secs, played, first_hit = 0, 0.0, 0, 0
    for t in range(MAX_TRIES):
        tries += 1
        secs += rng.lognormal(2.0, 0.5)
        if t == 0:
            hit = first_page_has_it(user, qtype, arm)
            first_hit = int(hit)
        else:
            hit = rng.random() < P_RETRY_HIT
        if rng.random() < (P_CLICK_HIT if hit else P_CLICK_MISS):
            played = int(rng.random() < P_PLAY)
            break
        if rng.random() > P_RETRY:
            break
    rows.append((arm, qtype, tries, secs, played, first_hit))

sessions = pd.DataFrame(rows, columns=["arm", "query_type", "attempts", "seconds",
                                       "played", "first_match"])

# A/B test on play conversion
control = sessions[sessions["arm"] == 0]
treat = sessions[sessions["arm"] == 1]
n1, n2 = len(control), len(treat)
p1, p2 = control["played"].mean(), treat["played"].mean()

_, srm_p = stats.chisquare([n1, n2])        # was the split even?
pooled = (control["played"].sum() + treat["played"].sum()) / (n1 + n2)
z = (p2 - p1) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
ci = [100 * ((p2 - p1) - 1.96 * se), 100 * ((p2 - p1) + 1.96 * se)]

# users needed per arm to see a 2 point change at 80% power
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
n_needed = int(np.ceil(2 * p1 * (1 - p1) * ((za + zb) / 0.02) ** 2))

ab = {
    "sessions": len(sessions),
    "srm_p_value": srm_p,
    "control_conversion": p1,
    "treatment_conversion": p2,
    "lift_points": 100 * (p2 - p1),
    "z": z,
    "p_value": p_value,
    "ci95_points": ci,
    "n_per_arm_for_2pt_lift": n_needed,
    "avg_attempts": sessions["attempts"].mean(),
    "avg_seconds": sessions["seconds"].mean(),
}
match_rates = (sessions.groupby(["query_type", "arm"])["first_match"].mean()
               .unstack().round(3))
print("\nA/B test")
print(json.dumps(ab, indent=2, default=float))
print("\nFirst-search match rate by query type (0 = control, 1 = treatment)")
print(match_rates)

ab["match_rate_by_query_type"] = match_rates.to_dict()
with open("results.json", "w") as f:
    json.dump({"offline": offline, "ab_test": ab}, f, indent=2, default=float)

# dashboard
grey, blue = "#9aa4b2", "#2563eb"
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].bar(["Popularity", "Item CF"],
               [offline["popularity"]["ndcg"], offline["item_cf"]["ndcg"]],
               color=[grey, blue])
axes[0, 0].set_title("Offline NDCG@10")

match_rates.reindex(QUERY_TYPES).plot.bar(ax=axes[0, 1], color=[grey, blue], legend=False)
axes[0, 1].set_title("First-search match rate (grey control, blue treatment)")
axes[0, 1].yaxis.set_major_formatter("{x:.0%}")
axes[0, 1].tick_params(axis="x", rotation=0)
axes[0, 1].set_xlabel("")

axes[1, 0].barh(["Played", "Search opened"], [100 * sessions["played"].mean(), 100], color=blue)
axes[1, 0].set_title("Funnel (% of sessions)")

tries_by_outcome = sessions.groupby("played")["attempts"].value_counts(normalize=True).unstack(0)
tries_by_outcome.plot.bar(ax=axes[1, 1], color=[grey, blue], legend=False)
axes[1, 1].set_title("Searches per session (grey closed, blue played)")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].set_xlabel("")

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)  