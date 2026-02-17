# Query batching helper

This repository includes helpers in `pyspark_batching_solution.py` for:
1. building `with_groups_df` from your embedding + LSH graph, and
2. batching grouped queries with a configurable `batch_size`.

## Build `with_groups_df`

You now have two options:

### Option A: pass `vec_df` + fitted `lsh_model`

```python
with_groups_df = build_with_groups_df(
    input_df=input_df,
    vec_df=vec_df,
    lsh_model=lsh_model,
    threshold=0.25,
    propagation_steps=6,
)
```

### Option B: build `vec_df` inside the function from `input_df`

```python
with_groups_df = build_with_groups_df(
    input_df=input_df,
    vec_df=None,
    lsh_model=None,
    embedding_fn=lambda text: model.encode(text).tolist(),
    threshold=0.25,
    propagation_steps=6,
    bucket_length=1.5,
    num_hash_tables=3,
)
```

Returned columns:
- `query`
- `location`
- `language`
- `final_group`

## Build final batches

```python
final_df = build_query_batches(with_groups_df, batch_size=5)
```

Output columns:
- `query` (array<string>)
- `location`
- `language`

Behavior:
- keeps similarity groups together,
- splits only groups larger than `batch_size`,
- packs chunks per `location` + `language`.

## Demo notebook

See `pyspark_batching_demo.ipynb` for an end-to-end example with a 50-row PySpark `input_df` that calls:
- `build_with_groups_df(...)`
- `build_query_batches(...)`
