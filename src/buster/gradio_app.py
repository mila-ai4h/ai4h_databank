from src.buster.buster_app import buster_app

buster_app.queue(concurrency_count=16, api_open=False)
buster_app.launch(share=False, show_api=False)
