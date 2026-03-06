from pyspark import StorageLevel
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType



def build_with_groups_df(input_df, model, lsh_model, threshold):
    """
    Build `with_groups_df` from `input_df` using embedding model + BRP-LSH.

    Parameters
    ----------
    input_df : pyspark.sql.DataFrame
        Must contain: query, location, language.
    model : object
        Embedding model with `.encode(text)` (for example SentenceTransformer).
    lsh_model : BucketedRandomProjectionLSH
        LSH estimator. If None, it is created with:
        inputCol='features', outputCol='hashes', bucketLength=1.5, numHashTables=3.
    threshold : float
        Distance threshold for approxSimilarityJoin (for example 0.25).

    Returns
    -------
    pyspark.sql.DataFrame
        Columns: query, location, language, final_group
    """

    embed_udf = F.udf(
        lambda text: [float(x) for x in model.encode(text).tolist()],
        ArrayType(FloatType()),
    )

    def _safe_normalize(v):
        if not v:
            return v
        norm = sum(i * i for i in v) ** 0.5
        if norm == 0.0:
            return [0.0 for _ in v]
        return [float(x) / norm for x in v]

    norm_udf = F.udf(_safe_normalize, ArrayType(FloatType()))
    to_vec_udf = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())

    prepped_df = input_df.withColumn(
        "query_compact",
        F.regexp_replace(F.col("query"), r"\s+", ""),
    )

    vec_df = (
        prepped_df
        .withColumn("embedding", embed_udf("query"))
        .withColumn("norm_vec", norm_udf("embedding"))
        .withColumn("features", to_vec_udf("norm_vec"))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    if lsh_model is None:
        lsh_model = BucketedRandomProjectionLSH(
            inputCol="features",
            outputCol="hashes",
            bucketLength=1.5,
            numHashTables=3,
        )

    fitted_lsh_model = lsh_model.fit(vec_df)

    similar_df = fitted_lsh_model.approxSimilarityJoin(
        vec_df,
        vec_df,
        threshold=threshold,
        distCol="distance",
    )

    pairs_df = (
        similar_df
        .filter(F.col("datasetA.query") < F.col("datasetB.query"))
        .select(
            F.col("datasetA.query").alias("query1"),
            F.col("datasetB.query").alias("query2"),
        )
    )

    edges_df = pairs_df.select("query1", "query2").unionByName(
        pairs_df.select(
            F.col("query2").alias("query1"),
            F.col("query1").alias("query2"),
        )
    )

    # Force links for space-variant forms (e.g. "iphone 15" and "iphone15").
    compact_links_df = (
        prepped_df
        .groupBy("query_compact")
        .agg(F.collect_set("query").alias("queries"))
        .where(F.size("queries") > 1)
        .select(F.explode("queries").alias("query1"), "queries")
        .select("query1", F.explode("queries").alias("query2"))
        .where(F.col("query1") != F.col("query2"))
    )

    edges_df = edges_df.unionByName(compact_links_df).distinct()

    vertices_df = input_df.select("query").distinct()
    clusters_df = vertices_df.withColumn("cluster_id", F.col("query"))

    for _ in range(6):
        propagated_df = (
            clusters_df
            .join(edges_df, clusters_df.query == edges_df.query1, "left")
            .select(
                clusters_df.query,
                F.least(
                    clusters_df.cluster_id,
                    F.coalesce(edges_df.query2, clusters_df.query),
                ).alias("new_cluster"),
            )
        )

        clusters_df = (
            propagated_df
            .groupBy("query")
            .agg(F.min("new_cluster").alias("cluster_id"))
        )

    result_df = (
        input_df
        .join(
            clusters_df.select(
                F.col("query").alias("query_key"),
                F.col("cluster_id").alias("final_group"),
            ),
            input_df.query == F.col("query_key"),
            "left",
        )
        .drop("query_key")
        .withColumn("final_group", F.coalesce(F.col("final_group"), F.col("query")))
        .select("query", "location", "language", "final_group")
    )

    vec_df.unpersist()
    return result_df


def build_query_batches(with_groups_df, batch_size=5):
    """
    Build query batches with a scale-first strategy.

    Strategy:
    - keep similarity groups intact by default,
    - split only oversized groups into chunks of size <= batch_size,
    - emit each chunk as its own batch row.

    This avoids expensive global packing/sorting across all groups and scales better
    when a project has a single `(location, language)` with very large row counts.
    """

    # Assign per-group row index so large groups can be sliced into fixed-size chunks.
    group_order_w = Window.partitionBy("final_group", "location", "language").orderBy(F.asc("query"))

    chunked_rows_df = (
        with_groups_df
        .withColumn("row_idx", F.row_number().over(group_order_w) - F.lit(1))
        .withColumn("chunk_id", (F.col("row_idx") / F.lit(batch_size)).cast("int"))
    )

    # Each (final_group, chunk_id, location, language) becomes one batch candidate.
    chunked_df = (
        chunked_rows_df
        .groupBy("final_group", "location", "language", "chunk_id")
        .agg(
            F.collect_list("query").alias("query"),
            F.count("*").alias("chunk_size"),
        )
    )

    # Deterministic batch numbering over chunk-level rows (much smaller than raw input rows).
    batch_w = Window.partitionBy("location", "language").orderBy(F.asc("final_group"), F.asc("chunk_id"))

    return (
        chunked_df
        .withColumn("batch_id", F.row_number().over(batch_w) - F.lit(1))
        .orderBy("location", "language", "batch_id")
        .select("query", "location", "language")
    )


def order_with_groups_df(with_groups_df):
    """
    Order grouped results by number of matches (desc), then group label (asc), then query (asc).

    Adds `match_count` for visibility and returns ordered rows.
    """

    group_sizes_df = (
        with_groups_df
        .groupBy("final_group", "location", "language")
        .agg(F.count("*").alias("match_count"))
    )

    return (
        with_groups_df
        .join(group_sizes_df, ["final_group", "location", "language"], "left")
        .orderBy(F.desc("match_count"), F.asc("final_group"), F.asc("query"))
    )
