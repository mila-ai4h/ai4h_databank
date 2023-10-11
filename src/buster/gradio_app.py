import os

from src.buster.buster_app import buster_app

# If no value was set, use default value
concurrency_count = int(os.getenv("CONCURRENCY_COUNT", 32))

buster_app.queue(concurrency_count=concurrency_count)
buster_app.launch(share=False)
