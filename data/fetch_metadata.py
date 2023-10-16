from buster.retriever import Retriever, ServiceRetriever
from src.cfg import buster_cfg

retriever: Retriever = ServiceRetriever(**buster_cfg.retriever_cfg)

# fetch all docs from all sources
df = retriever.get_documents()

# extract the metadata
metadata = df.groupby("url").apply(lambda x: x.iloc[0][["source", "title", "url", "country", "year"]])

# rename the columns
metadata = metadata.rename(
    columns={"source": "Source", "title": "Title", "year": "Year", "url": "Link", "country": "Country"}
)

# reorder columns
metadata = metadata[["Source", "Title", "Year", "Country", "Link"]]

# sort by source
metadata = metadata.sort_values("Source")

# save to csv
metadata.to_csv("documents_metadata.csv", index=False)
