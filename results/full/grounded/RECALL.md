# Question-grounded retrieval: recall of the answer evidence

613 answerable questions (933 in total), 82 sessions with cached 1-fps CLIP embeddings. Object phrase pulled out of 883/933 questions (the rest use the full question). 4 answerable questions have evidence in a session without cached embeddings (counted as misses).

**Recall@X** = % of answerable questions with an evidence span (+-2 s) inside one of the first X windows (anchor +- W s). Random = random anchors with the same spacing, averaged over seeds.

## sessions = evidence, frames searched = all, window = +-15 s (median 1930 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 14.4 | 21.9 | 32.6 | 48.6 |
| object | 12.2 | 21.4 | 34.9 | 52.2 |
| prompt | 15.5 | 23.5 | 35.6 | 53.0 |
| random | 3.6 | 6.9 | 13.5 | 25.3 |

## sessions = evidence, frames searched = all, window = +-30 s (median 1930 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 17.3 | 26.1 | 39.5 | 58.6 |
| object | 16.3 | 26.8 | 40.3 | 59.5 |
| prompt | 18.6 | 27.9 | 40.5 | 58.7 |
| random | 4.8 | 9.4 | 18.8 | 35.1 |

## sessions = evidence, frames searched = uniform100, window = +-15 s (median 100 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 10.1 | 17.1 | 27.9 | 40.9 |
| object | 7.0 | 15.8 | 24.5 | 40.9 |
| prompt | 11.9 | 19.7 | 28.7 | 44.4 |
| random | 2.9 | 5.9 | 12.8 | 25.8 |

## sessions = evidence, frames searched = uniform100, window = +-30 s (median 100 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 12.7 | 20.9 | 33.4 | 51.9 |
| object | 9.3 | 19.1 | 29.7 | 49.8 |
| prompt | 14.0 | 22.5 | 33.4 | 53.2 |
| random | 4.5 | 8.8 | 18.8 | 36.3 |

## sessions = prior, frames searched = all, window = +-15 s (median 11059 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 5.7 | 8.5 | 13.1 | 25.3 |
| object | 5.2 | 7.3 | 13.7 | 24.8 |
| prompt | 5.5 | 8.3 | 15.8 | 25.4 |
| random | 0.9 | 1.7 | 3.2 | 6.8 |

## sessions = prior, frames searched = all, window = +-30 s (median 11059 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 7.5 | 10.8 | 17.1 | 30.8 |
| object | 7.3 | 10.1 | 17.5 | 29.2 |
| prompt | 7.0 | 11.3 | 19.4 | 29.4 |
| random | 1.3 | 2.4 | 4.7 | 10.2 |

## sessions = prior, frames searched = uniform100, window = +-15 s (median 458 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 3.3 | 7.5 | 11.7 | 18.8 |
| object | 2.4 | 6.2 | 10.0 | 15.5 |
| prompt | 3.1 | 7.2 | 10.9 | 16.6 |
| random | 1.1 | 2.2 | 4.0 | 7.1 |

## sessions = prior, frames searched = uniform100, window = +-30 s (median 458 frames per question)

| query | X=1 | X=2 | X=4 | X=8 |
|---|---|---|---|---|
| question | 4.7 | 9.5 | 14.7 | 24.6 |
| object | 3.6 | 8.0 | 12.4 | 19.9 |
| prompt | 4.4 | 9.0 | 13.5 | 20.9 |
| random | 1.4 | 3.2 | 5.6 | 10.5 |

Default setting {'sessions': 'evidence', 'pool': 'all', 'query': 'prompt', 'w': 15}: evidence found by one of the 8 windows for 53.0% of answerable questions; median rank of the first hit 3.
