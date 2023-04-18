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
export AI4H_USERNAME=
export AI4H_PASSWORD=
export OPENAI_API_KEY=
export OPENAI_ORGANIZATION=
export HUB_TOKEN=
export MONGODB_AI4H_PASSWORD=
export MONGODB_AI4H_USERNAME=
export MONGODB_AI4H_DB_NAME=

cd src
gradio gradio_app.py
```

## How-to fill out a database

1. Create a parser that generates a dataframe of chunks.
2. Create a `DocumentsManager` from `buster`.
3. Send the chunks to the `DocumentsManager` by using `buster.docparser.generate_embeddings`.
