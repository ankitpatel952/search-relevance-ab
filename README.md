# Search relevance and A/B test simulation

I wanted to check one thing: if a better ranking model scores higher offline, does that show up in conversion? I compared a popularity ranker with item-based collaborative filtering on real movie ratings, then ran a simulated A/B test on top.

![Dashboard](dashboard.png)

## Data

MovieTweetings (github.com/sidooms/MovieTweetings), about 921k ratings on a 0-10 scale. I kept the 3,000 most rated movies and users with at least 20 ratings, which leaves 8,140 users and 535,276 ratings. A rating of 8 or more counts as "liked". The data folder is not in this repo, so download ratings.dat and movies.dat from the latest folder and put them in data/.

## Running it

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python search_relevance_ab.py data/

It takes a few minutes and writes results.json and dashboard.png.

## What the script does

- Splits each user's ratings by time. The latest 20% are the test set.
- Ranks movies two ways: by popularity, and by item-item cosine similarity.
- Scores both at K=10 with precision, NDCG, MRR and hit rate.
- Simulates 20,000 search sessions with four query types (exact title, genre, vague, misspelled), split evenly between the two rankers.
- Tests the difference in play conversion with a two-proportion z-test, a 95% confidence interval and a sample ratio check.

## Results

| | Popularity | Item CF |
|---|---|---|
| Precision@10 | 1.82% | 2.38% |
| NDCG@10 | 0.0307 | 0.0393 |
| MRR | 0.0538 | 0.0624 |
| Hit rate@10 | 14.2% | 18.3% |

NDCG went up about 28%. In the simulated test, conversion was 62.78% for control and 62.59% for treatment (p = 0.78, 95% CI -1.53 to +1.15 points), so no measurable change.

## What I took from it

Only genre queries depend on the ranker here, and the match rate on those went from 11.2% to 13.2%. Exact and misspelled queries are matched by title, and vague queries barely moved. Those effects are too small to show up in overall conversion. A better offline score doesn't automatically mean a better product metric, and it depends on how much of the traffic the model actually touches.

## Limits

- The sessions are simulated. The click and play probabilities are my assumptions, set at the top of the script, so the A/B result shows the method and says nothing about real users.
- The offline metrics are measured on real ratings.
- Only one split and one seed (42) were run.