from src.buster.buster_app import buster_app

buster_app.queue(concurrency_count=16)
buster_app.launch(share=False)