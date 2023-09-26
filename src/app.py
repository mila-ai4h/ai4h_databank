import gradio as gr
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from app_utils import check_auth
from src.arena.arena_app import arena_app
from src.buster.buster_app import buster_app

app = FastAPI()


def add_auth(gradio_app):
    gradio_app.auth = check_auth
    gradio_app.auth_message = ""
    gradio_app.queue(concurrency_count=16)
    return gradio_app


buster_app = add_auth(buster_app)
arena_app = add_auth(arena_app)

app = gr.mount_gradio_app(app, buster_app, path="/buster")
app = gr.mount_gradio_app(app, arena_app, path="/arena")


@app.get("/")
# redirects to the buster page when initially visiting the website
async def root():
    return RedirectResponse(url="/buster")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", reload=True)
