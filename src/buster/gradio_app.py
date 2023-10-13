import os
from src.buster.buster_app import buster_app

concurrency_count = int(os.getenv("CONCURRENCY_COUNT", 32))

buster_app.queue(concurrency_count=concurrency_count, api_open=False)
buster_app.launch(share=False, show_api=False)
