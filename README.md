# Query batching helper

This repository includes helpers in `pyspark_batching_solution.py` for:
1. building `with_groups_df` from `input_df`, and
2. ordering grouped queries by match volume + alphabetic group name, and
3. batching grouped queries with a configurable `batch_size`.

## Build `with_groups_df`

`build_with_groups_df` uses exactly these parameters:
- `input_df`
- `model`
- `lsh_model`
- `threshold`

```python
from sentence_transformers import SentenceTransformer
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark_batching_solution import build_with_groups_df

model = SentenceTransformer("all-MiniLM-L6-v2")

lsh_model = BucketedRandomProjectionLSH(
    inputCol="features",
    outputCol="hashes",
    bucketLength=1.5,
    numHashTables=3,
)

with_groups_df = build_with_groups_df(
    input_df=input_df,
    model=model,
    lsh_model=lsh_model,
    threshold=0.25,
)
```

Notes:
- `vec_df` is built internally from `input_df`.
- if `lsh_model` is passed as `None`, the same BRP-LSH config above is created internally.
- space-variant forms are explicitly linked (e.g., `iphone 15` and `iphone15`).
- safe normalization is used to avoid division-by-zero for zero vectors.

Returned columns:
- `query`
- `location`
- `language`
- `final_group`

## Build final batches

```python
from pyspark_batching_solution import build_query_batches

final_df = build_query_batches(with_groups_df, batch_size=5)
```

Output columns:
- `query` (array<string>)
- `location`
- `language`

Behavior:
- keeps similarity groups together,
- splits only groups larger than `batch_size`,
- packs chunks per `location` + `language` using strict greedy chunk packing.
- never exceeds `batch_size` in any output row.

## Demo notebook

See `pyspark_batching_demo.ipynb` for an end-to-end example with a 50-row PySpark `input_df` calling both functions.


## Order grouped output

```python
from pyspark_batching_solution import order_with_groups_df

ordered_with_groups_df = order_with_groups_df(with_groups_df)
# columns include: query, location, language, final_group, match_count
# order: match_count desc, final_group asc, query asc
```
