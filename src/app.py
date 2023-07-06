import gradio as gr
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from buster_app import buster_app

app = FastAPI()

app = gr.mount_gradio_app(app, buster_app, path="/buster")


@app.get("/")
# redirects to the buster page when initially visiting the website
async def root():
    return RedirectResponse(url="/buster")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)