# SAI ️💬

SAI ️💬 is a Q&A search engine designed to provide relevant and high quality information about curated AI policy documents.

This project is a collaboration between the OECD and Mila.

It uses Retrieval-Augmented Generation (RAG) on AI policy documents curated by the OECD.


## Hosting
<!-- It deploys [buster](www.github.com/jerpint/buster) on AI policies collected by the OECD. -->

Links to current deployments of the app. We host the app on huggingface as well as on heroku.

| Service       | Dev URL                                                                                          | Prod URL                                                                                             |
|---------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Heroku        | [Dev](https://ai4h-databank-dev.herokuapp.com/)                                                 | [Prod](https://ai4h-databank-prod.herokuapp.com/)                                                   |
| Huggingface   | [Dev (private)](https://huggingface.co/spaces/mila-quebec/SAI-dev)                              | [Prod (public)](https://huggingface.co/spaces/mila-quebec/SAI)                                     |

Note that the Dev space on huggingface is private and you need to be a member of the org. to view it.
Note that on Heroku, a username and password are required to sign in:

```
username: databank-$USERNAME
password: MilaDatabank!!123
```

Where `$USERNAME` can be any username, ideally used to identify who is using the app (e.g. `databank-jeremy`)


## Tech Stack

Here is an overview of our tech stack:

![image](assets/tech_stack.png)


## How to run locally

### Install the dependencies

It is recommended to work in a virtual environment (e.g. conda) when running locally.
Simply clone the repo and install the dependencies.

Or, in a terminal:
```sh
git clone git@github.com:mila-iqia/ai4h_databank.git
cd ai4h_databank

# install the package locally
pip install -e .
```


Note that SAI requires python>=3.10


### Environment variables

The app relies on configured environment variables for authentication to openai, mongodb, and pinecone as well as some server information:
You will first need to configure the environment variables:

```sh
export OPENAI_ORGANIZATION=...
export OPENAI_API_KEY=sk-...
export MONGO_URI=...
export PINECONE_API_KEY=...
export INSTANCE_TYPE= ... # One of [dev, prod, local]. Determines where to log all app interactions See the logging section for more details.
export INSTANCE_NAME= ... # An identifier to know which platform we are logging from (e.g. huggingface-server-1). Stored as metadata when collecting user interactions.
```

To get access to the secrets, contact the app maintainers.

Note that if any of the environment variables are missing, the app might not launch.

### Running the app

There are currently 2 ways of running the gradio apps, via `gradio` or as a mounted app.

Note that in the context of this project we have 2 separate apps, the main buster app (SAI) and a separate arena app to evaluate different models and parameters in a blind test. We present 2 different options to run the apps here:

#### Gradio (Recommended)

In this setup, only one app is run at a time.
Go to the folder of the app and simply run the app from there:
This setup is recommended for deploying on huggingface.

To launch the SAI app:
```sh
cd src/buster
gradio gradio_app.py buster_app
```

This will launch the SAI app locally. Then go to the localhost link to see the deployed app.

To launch the arena app:
```sh
cd src/arena
gradio gradio_app.py arena_app
```


#### Mounted app

When the app is mounted, it allows for both arena and buster apps to run simultaneously.
This type of deployment is only supported on heroku, not on huggingface.
Using the mounted app approach allows a user to deploy the app with authentication and have 2 separate pages, `/buster` and `/arena`.

```sh
cd src/
python app.py
```

Note that the mounted app launches with authentication enabled. To set usernames and passwords, set environment variables:

```sh
export AI4H_APP_USERNAME=...
export AI4H_APP_PASSWORD=...
```

Also note that the auth. uses gradio's built-in authentication and is currently set such that any username starting with `AI4H_APP_USERNAME` will be considered valid. See `app_utils.py:check_auth`.

### App Deployment

#### CI/CD

The simplest way to deploy the apps is via the CI/CD pipelines. We have automated deployment using github actions.

Every time a new PR is opened, the CI/CD runs and deploys the apps on dev instances once all checks pass.
Checks include unit tests and linting (black and isort).

Once a PR gets merged to `main`, the app is then deployed to the `prod` instances.

Currently, the CI/CD pipeline deploys the mounted app on heroku and the buster gradio app on huggingface.

#### Manual Deployment

To manually deploy the app to huggingface spaces, you can use the script in `scripts/deploy_hf_space.sh`. For example, to deploy the app in `dev`, run:

```bash
sh scripts/deploy_hf_space.sh dev
```


## Configuring the App

The app uses [buster 🤖](www.github.com/jerpint/buster) to power its main features.

Buster relies on a config file which initializes the environments and loads all necessary features. This config file is found in `src/cfg.py`, where all of the settings including models used, prompts, number of sources retrieved, etc. can be tuned. Typically, if a setting needs to be changed, it should be changed in this file.

Refer to Buster documentation to learn more about customizing and adding features.

The frontend is all powered by Gradio's interface. The app layout is mainly found in `src/buster/buster_app.py`. Note that app development was done using gradio v3. During the latest stages of the app, gradio v4 was released, however it is not backwards compatible and introduced breaking changes in our app. We recommend for the time being to stick to the gradio version pinned in the `requirements.txt` version.

## Chunk Management

We built our own chunk management service, which combines both pinecone and mongoDB. Pinecone is used exclusively as a vector store, and all metadata associated to vectors (contents, year, links, etc.) are indexed in mongodb.

### Uploading Vectors to Pinecone

@hbertrand TODO

### Uploading Documents to MongoDB

@hbertrand TODO

## Logging and Feedback

### Collections

The app supports collecting user feedback and interaction, which is all stored on a cloud instance of mongodb.

The `INSTANCE_TYPE` env. variable determines which database to log to (one of `[prod, dev, or local]`), so that we separate the logs from the different environments.

We record 3 different types of user interactions:

* `interaction`: After a user asks a question and SAI answers it, both the user's question, SAI's answer, cited sources, and other relevant information (parameters, models, etc.) are collected in MongoDB in the `interaction` collection.
* `feedback`: A feedback form is available for users to fill in the app. Every time a user submits a feedback form, the entire app state is recorded and logged in mongodb. This includes the form filled out by the user, as well as the question asked, the sources, the response, parameters, etc.
* `flagged`: A flagged button is available on the app to monitor for any kind of harmful content. This records the app state in a separate mongo table.

Note that these collection names, where they are logged, etc. can all be adjusted in the `src/cfg.py` file.

### Retrieving user interactions

We have created some useful scripts to download all user interactions and feedback. This will dump them all to .csv files for easier handling and analysis afterwards.

For example, to dump user interactions from a specific date onwards, simply use:

```bash
python scripts/dump_collections.py interaction
```

### System logs

We use python's builtin `logging` module to handle the app logging.

On huggingface, system logs are ephemeral. You can view the logs in real time directly from the HF space dashboard. Every time the app is reset, the systems logs are lost. The logging also gets reset every time a new build is triggered.

On heroku, we use papertrail to capture and store system logs. This needs to be setup for the app directly on heroku using their plugin store.

## How to backup a database

```sh
mongodump --archive="backup-ai4h-databank-prod-2023-09-29" --db="ai4h-databank-prod" --uri="mongodb+srv://ai4h-databank-dev.m0zt2w2.mongodb.net/" --username miladatabank
mongorestore --archive="backup-ai4h-databank-prod-2023-09-29" --nsFrom='ai4h-databank-prod.*' --nsTo='backup-ai4h-databank-prod-2023-09-29.*' --uri="mongodb+srv://ai4h-databank-dev.m0zt2w2.mongodb.net/" --username miladatabank
```