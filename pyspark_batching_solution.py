from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType


def build_with_groups_df(
    input_df,
    vec_df=None,
    lsh_model=None,
    threshold=0.25,
    propagation_steps=6,
    embedding_fn=None,
    bucket_length=1.5,
    num_hash_tables=3,
):
    """
    Build `with_groups_df` from a similarity graph.

    You can pass either:
    1) precomputed `vec_df` + fitted `lsh_model`, or
    2) only `input_df` and `embedding_fn` (this function will build vec_df and fit LSH).

    Parameters
    ----------
    input_df : pyspark.sql.DataFrame
        Must contain: query, location, language.
    vec_df : pyspark.sql.DataFrame, optional
        Must contain: query, features. If missing, it is derived from `input_df`.
    lsh_model : BucketedRandomProjectionLSHModel, optional
        Fitted model. If missing, it is fit on vec_df.
    threshold : float
        Distance threshold used in approxSimilarityJoin.
    propagation_steps : int
        Number of label-propagation iterations.
    embedding_fn : callable, optional
        Function text -> list[float], required when vec_df is not provided.
    bucket_length : float
        LSH bucket length when fitting inside this function.
    num_hash_tables : int
        Number of LSH hash tables when fitting inside this function.

    Returns
    -------
    pyspark.sql.DataFrame
        Columns: query, location, language, final_group
    """

    if vec_df is None:
        if embedding_fn is None:
            raise ValueError(
                "When vec_df is None, you must provide embedding_fn(text) -> list[float]."
            )

        embed_udf = F.udf(lambda text: embedding_fn(text), ArrayType(FloatType()))
        def _safe_normalize(v):
            if not v:
                return v
            norm = sum(i * i for i in v) ** 0.5
            if norm == 0.0:
                return [0.0 for _ in v]
            return [float(x) / norm for x in v]

        norm_udf = F.udf(_safe_normalize, ArrayType(FloatType()))
        to_vec_udf = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())

        vec_df = (
            input_df
            .withColumn("embedding", embed_udf("query"))
            .withColumn("norm_vec", norm_udf("embedding"))
            .withColumn("features", to_vec_udf("norm_vec"))
        )

    if lsh_model is None:
        lsh = BucketedRandomProjectionLSH(
            inputCol="features",
            outputCol="hashes",
            bucketLength=bucket_length,
            numHashTables=num_hash_tables,
        )
        lsh_model = lsh.fit(vec_df)

    similar_df = lsh_model.approxSimilarityJoin(
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

    vertices_df = input_df.select("query").distinct()
    clusters_df = vertices_df.withColumn("cluster_id", F.col("query"))

    for _ in range(propagation_steps):
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

    return (
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


def build_query_batches(with_groups_df, batch_size=5):
    """Build query batches while preserving similarity groups."""

    groups_df = (
        with_groups_df
        .groupBy("final_group", "location", "language")
        .agg(
            F.collect_list("query").alias("queries"),
            F.count("*").alias("group_size"),
        )
    )

    exploded_df = groups_df.select(
        "final_group",
        "location",
        "language",
        F.posexplode("queries").alias("pos", "term"),
    )

    chunked_df = (
        exploded_df
        .withColumn("chunk_id", (F.col("pos") / F.lit(batch_size)).cast("int"))
        .groupBy("final_group", "location", "language", "chunk_id")
        .agg(
            F.collect_list("term").alias("queries"),
            F.count("*").alias("chunk_size"),
        )
    )

    order_w = Window.partitionBy("location", "language").orderBy(
        F.desc("chunk_size"), "final_group", "chunk_id"
    )

    packed_df = (
        chunked_df
        .withColumn(
            "running_size",
            F.sum("chunk_size").over(
                order_w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            ),
        )
        .withColumn(
            "batch_id",
            ((F.col("running_size") - 1) / F.lit(batch_size)).cast("int"),
        )
    )

    return (
        packed_df
        .groupBy("batch_id", "location", "language")
        .agg(F.flatten(F.collect_list("queries")).alias("query"))
        .orderBy("batch_id")
        .select("query", "location", "language")
    )
