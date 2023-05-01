import logging

import pytest
from buster.busterbot import Buster
from buster.utils import get_retriever_from_extension

from src import cfg

DB_FILE = "../documents_oecd.db"


@pytest.fixture
def busterbot(monkeypatch, run_expensive):
    if not run_expensive:
        # Patch embedding call
        monkeypatch.setattr(Buster, "get_embedding", lambda s, x, engine: [0.0] * 1536)

    retriever = get_retriever_from_extension(DB_FILE)(DB_FILE)
    buster = Buster(cfg=cfg.buster_cfg, retriever=retriever)
    return buster


def test_irrelevant_questions(busterbot):
    logging.info(busterbot.unk_embedding)
    print("here")
