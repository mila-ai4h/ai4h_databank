from src.arena.arena_app import arena_app

arena_app.queue(concurrency_count=16)
arena_app.launch(share=False)