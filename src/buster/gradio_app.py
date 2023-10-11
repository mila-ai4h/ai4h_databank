import os

from src.buster.buster_app import buster_app

buster_app.queue(concurrency_count=os.getenv("CONCURRENCY_COUNT", 16))
buster_app.launch(share=False)
