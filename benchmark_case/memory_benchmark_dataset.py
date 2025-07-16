import datasets

DS_PATH = "/mnt/data/gyzou/expr_workplace/self-agent-memory/benchmark_cache/manual_hf_download/MemoryAgentBench"


def get_ttl_ds():
    ds = datasets.load_dataset(DS_PATH)
    ttl_ds = ds["Test_Time_Learning"]
    ttl_pd = ttl_ds.to_pandas()
    return ttl_pd.iloc[1:].to_dict(orient="records")
