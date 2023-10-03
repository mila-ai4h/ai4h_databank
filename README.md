# AI4H - Databank

This project is a collaboration between the OECD and Mila. It deploys [buster](www.github.com/jerpint/buster) on AI policies collected by the OECD.

## Deployments

Links to current deployments of the app. We host the app on huggingface as well as on heroku.

### Heroku

Auth. is required on heroku. Use the following for authentication. Any username that begins with `databank-` is considered valid:

- Username: databank-$USERNAME
- Password: MilaDatabank!!123

Dev: [https://ai4h-databank-dev.herokuapp.com/](https://ai4h-databank-dev.herokuapp.com/)

Prod: [https://ai4h-databank-prod.herokuapp.com/](https://ai4h-databank-prod.herokuapp.com/)

### Huggingface

No auth. required on Huggingface.

Dev:
[https://huggingface.co/spaces/databank-ai4h/buster-dev](https://huggingface.co/spaces/databank-ai4h/buster-dev)

Prod: TODO: Update when prod is officially live.


## How-to install

It is recommended to work in a virtual environment (e.g. conda) when running locally.
Simply clone the repo and install the dependencies.

Or, in a terminal:
```sh
git clone git@github.com:mila-iqia/ai4h_databank.git
cd ai4h_databank
pip install -r requirements.txt
```


Note that buster requires python>=3.10

## How-to run

### Environment variables

The app relies on configured environment variables for authentication to openai, mongodb, and pinecone as well as some server information:
You will first need to configure the environment variables.

Or, in a terminal:
```sh
export OPENAI_ORGANIZATION=...
export OPENAI_API_KEY=sk-...
export MONGO_USERNAME=...
export MONGO_PASSWORD=...
export MONGO_CLUSTER=...
export PINECONE_API_KEY=...
export INSTANCE_TYPE= ... # One of [dev, prod, local]
export INSTANCE_NAME= ... # An identifier to know which platform we are running on (e.g. huggingface-server-1)
```

To get access to the secrets, contact the app maintainers.

### Running the app

There are currently 2 ways of running the app, via gradio or as a mounted app.
When the app is mounted, it allows for multiple endpoints to be exposed.

#### Gradio

Go to the folder of the app and simply run the app from there:

```sh
cd src/buster
gradio gradio_app.py
```

#### Mounted app

simply run

```sh
cd src/
python app.py
```

Note that the mounted app launches with authentication enabled. To set usernames and passwords, simply:

```sh
export AI4H_APP_USERNAME=...
export AI4H_APP_PASSWORD=...
```

Note that the auth. uses gradio's built-in authentication and is currently set such that any username starting with `AI4H_APP_USERNAME` will be considered valid. See `app_utils.py:check_auth`.

### App Deployment

The simplest way to deploy the apps is via the CI/CD pipelines.

Every time a new PR is opened, the CI/CD runs and deploys the apps on dev instances (assuming all checks pass)

Once merged to `main`, the app is then deployed to the `prod` instances.

Currently, the CI/CD pipeline deploys the mounted app on heroku and the buster gradio app on huggingface.

## Chunk Management

TODO: Add details on how to update/manage chunks


## How to backup a database

```sh
mongodump --archive="backup-ai4h-databank-prod-2023-09-29" --db="ai4h-databank-prod" --uri="mongodb+srv://ai4h-databank-dev.m0zt2w2.mongodb.net/" --username miladatabank
mongorestore --archive="backup-ai4h-databank-prod-2023-09-29" --nsFrom='ai4h-databank-prod.*' --nsTo='backup-ai4h-databank-prod-2023-09-29.*' --uri="mongodb+srv://ai4h-databank-dev.m0zt2w2.mongodb.net/" --username miladatabank
```