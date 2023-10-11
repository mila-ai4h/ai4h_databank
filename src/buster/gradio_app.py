import os
from src.buster.buster_app import buster_app

# Allows the concurrency count to be set as an env. variable, otherwise default to 32.
concurrency_count = int(os.getenv("CONCURRENCY_COUNT", 32))

buster_app.queue(concurrency_count=concurrency_count)
buster_app.launch(share=False)
