import os

from src.buster.buster_app import buster_app

if __name__ == "__main__":
    buster_app.queue(api_open=False)
    buster_app.launch(share=False, show_api=False)
