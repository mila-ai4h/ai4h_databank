# ai4h_databank

## Links

Dev:
- [https://ai4h-databank-dev.herokuapp.com/](https://ai4h-databank-dev.herokuapp.com/)
- Username: databank-test
- Password: MilaDatabank!!123

Prod:
- [https://ai4h-databank-prod.herokuapp.com/](https://ai4h-databank-prod.herokuapp.com/)
- Username: databank-test
- Password: MilaDatabank!!123


## How-to install

1. Clone the repo
2. Create an env
3. Install all the dependencies

Or, in a terminal:
```sh
git clone git@github.com:mila-iqia/ai4h_databank.git
cd ai4h_databank
conda create -n databank python
conda activate databank
pip install -e .
```

## How-to run

1. Create the needed env variables.
2. Launch with Gradio

Or, in a terminal:
```sh
export OPENAI_ORGANIZATION=...
export AI4H_USERNAME=...
export AI4H_PASSWORD=...
export AI4H_MONGODB_USERNAME=...
export AI4H_MONGODB_PASSWORD=...
export AI4H_MONGODB_DB_NAME=...
export AI4H_MONGODB_CLUSTER=...
export AI4H_PINECONE_API_KEY=...
export AI4H_PINECONE_ENV=...
export AI4H_PINECONE_INDEX=...
export AI4H_MONGODB_DB_DATA=...
export AI4H_MONGODB_FEEDBACK_COLLECTION=...

cd src
gradio gradio_app.py
```

## How-to fill out a database

1. Create a parser that generates a dataframe of chunks.
2. Create a `DocumentsManager` from `buster`.
3. Send the chunks to the `DocumentsManager` by using `buster.docparser.generate_embeddings`.
