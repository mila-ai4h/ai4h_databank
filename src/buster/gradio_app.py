import os

from src.buster.buster_app import buster_app

# If no value was set, use default value
concurrent_count = int(os.getenv("CONCURRENCY_COUNT", 16))

buster_app.queue(concurrency_count=concurrent_count)
buster_app.launch(share=False)
