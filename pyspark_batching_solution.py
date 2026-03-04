from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructField, StructType
import pandas as pd



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


def _assign_batch_ids(sizes, batch_size):
    """Greedy contiguous assignment that never lets a batch exceed batch_size."""
    batch_ids = []
    current_batch = 0
    current_size = 0

    for size in sizes:
        size = int(size)
        if current_size > 0 and current_size + size > int(batch_size):
            current_batch += 1
            current_size = 0
        batch_ids.append(current_batch)
        current_size += size

    return batch_ids


def build_query_batches(with_groups_df, batch_size=5):
    """Build query batches while preserving similarity groups and strictly enforcing max batch size."""

    groups_df = (
        with_groups_df
        .groupBy("final_group", "location", "language")
        .agg(
            F.collect_list("query").alias("queries"),
            F.count("*").alias("group_size"),
        )
    )

    # Split only oversized groups into fixed chunks (chunk_size <= batch_size).
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

    # applyInPandas requires an explicit output schema, so we declare it up front.
    # We keep the original chunk metadata and add `batch_id` which is assigned
    # inside each (location, language) partition.
    assign_schema = StructType([
        StructField("location", chunked_df.schema["location"].dataType, False),
        StructField("language", chunked_df.schema["language"].dataType, False),
        StructField("final_group", chunked_df.schema["final_group"].dataType, False),
        StructField("chunk_id", IntegerType(), False),
        StructField("queries", chunked_df.schema["queries"].dataType, False),
        StructField("chunk_size", IntegerType(), False),
        StructField("batch_id", IntegerType(), False),
    ])

    def _pack_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        # Sort chunks so packing is deterministic:
        # - bigger chunks first (better fill efficiency),
        # - then stable tie-breakers by group/chunk id.
        pdf = pdf.sort_values(["chunk_size", "final_group", "chunk_id"], ascending=[False, True, True]).reset_index(drop=True)

        # Greedy assignment: start a new batch only when adding the next chunk
        # would exceed `batch_size`. This guarantees each batch stays within limit.
        pdf["batch_id"] = _assign_batch_ids(pdf["chunk_size"].tolist(), batch_size)

        # Enforce integer dtypes expected by Spark for schema consistency.
        pdf["chunk_id"] = pdf["chunk_id"].astype(int)
        pdf["chunk_size"] = pdf["chunk_size"].astype(int)
        pdf["batch_id"] = pdf["batch_id"].astype(int)

        # Return exactly the columns declared in `assign_schema`.
        return pdf[["location", "language", "final_group", "chunk_id", "queries", "chunk_size", "batch_id"]]

    # Execute the packing per locale partition so each project's language/location
    # is packed independently and receives its own batch_id sequence.
    packed_df = chunked_df.groupBy("location", "language").applyInPandas(_pack_partition, schema=assign_schema)

    return (
        packed_df
        .groupBy("batch_id", "location", "language")
        .agg(F.flatten(F.collect_list("queries")).alias("query"))
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
