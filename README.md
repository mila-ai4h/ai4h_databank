# AIR ️💬

AIR ️💬 is a Q&A search engine designed to provide relevant and high quality information about curated AI policy documents.

This project is a collaboration between the OECD and Mila.

It uses Retrieval-Augmented Generation (RAG) on AI policy documents curated by the OECD.


## Hosting
<!-- It deploys [buster](www.github.com/jerpint/buster) on AI policies collected by the OECD. -->

Links to current deployments of the app. We host the app on huggingface.

| Service       | Dev URL                                                                                          | Prod URL                                                                                             |
|---------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Huggingface   | [Dev (private)](https://huggingface.co/spaces/mila-ai4h/SAI-dev)                              | [Prod (public)](https://huggingface.co/spaces/mila-ai4h/SAI)                                     |

Note that the Dev space on huggingface is private and you need to be a member of the org. to view it.


## Tech Stack

Here is an overview of our tech stack:

- Frontend: Gradio
- Documents store: Pinecone for the vectors, MongoDB for the text
- Backend: OpenAI API + [Buster](https://github.com/jerpint/buster)
- Deployment: Huggingface
- CI/CD: GitHub Actions
- Logging: MongoDB


## How to run locally

### Install the dependencies

It is recommended to work in a virtual environment (e.g. conda) when running locally.
Simply clone the repo and install the dependencies.

Or, in a terminal:
```sh
git clone git@github.com:mila-ai4h/ai4h_databank.git
cd ai4h_databank

# install the package locally
pip install -e .
```


Note that AIR requires python>=3.10


### Environment variables

The app relies on configured environment variables for authentication to openai, mongodb, and pinecone as well as some server information:
You will first need to configure the environment variables:

```sh
export OPENAI_ORG_ID=...
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

To launch the AIR app:
```sh
cd src/buster
gradio gradio_app.py buster_app
```

This will launch the AIR app locally. Then go to the localhost link to see the deployed app.

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

## Data Management

AIR requires two services to store the data. MongoDB is used to store the documents as well as their associated metadata (year, country, link, ...). Pinecone is a vector store and is meant only for storing the embeddings associated with each chunk, as well as an identifier to link it back to the correct document in MongoDB.

The rest of the data (logging, interactions, feedback) is detailed in the section [Logging and Feedback](#logging-and-feedback).

### Uploading documents

Uploading documents is done through the script `script/upload_data.py`. The script expects the name of the MongoDB database to use, the name of the Pinecone namespace to use, and one or more files. It is highly recommended to use the same name for the MongoDB database and the Pinecone namespace, for easier versioning. The current convention is `data-YYYY-MM-DD`.

The files should be CSV files with tab delimiters. Each row should be a chunk, of no more than 1000 tokens. If some chunks are bigger than 1000 tokens, they will be cut in smaller chunks at the token level.

The number of words that fit in 1000 tokens depends on the alphabet and the language. For english, 1000 tokens is around 1500 words. For other languages using the latin alphabet, 1000 tokens is often a bit less than 1500 words. Non-latin alphabets have very different limits.

The token limit can be changed with the `--token_limit_per_chunk` argument.

The minimum expected columns of the files are: content, url, title, source, country, year. They are required because they are used in various ways throughout SAI. Additional columns will be stored as metadata in MongoDB, but ignored otherwise. An example of a valid file is provided in `data/example_chunks.csv`.

The process of uploading documents is as follows:
- Check that all chunks are less than 500 tokens, and cut them if necessary.
- Check that all required columns are present.
- Compute the embeddings.
- Upload all the chunks to MongoDB, and retrieve their unique identifiers.
- Upload all the embeddings to Pinecone, with their MongoDB identifiers.

### Deleting documents

Deleting documents is done through the script `scripts/delete_data.py`. The script expects the name of the MongoDB database to use, the name of the Pinecone namespace to use, and either a `--source` argument followed by a name, or the `--all` argument.

If specifying a source, all documents from that source will be deleted, in both MongoDB and Pinecone.

If deleting all documents, the specified Pinecone namespace will be deleted. The specified MongoDB database cannot be deleted automatically through a normal API key, and must be manually deleted on the web UI. The script will remind you of that point.

### Example on how to upload one file, then delete the documents

You will need to have setup the environment properly ([installed the dependencies](#install-the-dependencies) and [setup the environment variables](#environment-variables)).

First, let's upload the example chunks in the Pinecone namespace `data-example` and the MongoDB database `data-example`:

```sh
python scripts/upload_data.py data-example data-example "data/example_chunks.csv"
```

If we now want to delete those documents, we can do:
```sh
python scripts/delete_data.py data-example data-example --all
```

This will delete the Pinecone namespace `data-example`. The MongoDB database `data-example` needs to be dropped manually from the web UI.

### Switching to a new version of the data

To change the version of the data that is being used, specify the desired `PINECONE_NAMESPACE` and `MONGO_DATABASE_DATA` in `src/cfg.py`.

If `PINECONE_NAMESPACE` and `MONGO_DATABASE_DATA` are not identical, a warning is raised when launching the app.


### How to backup a database

Useful commands to make a local backup of a database:

```sh
mongodump --archive="backup-ai4h-databank-prod-2023-09-29" --db="ai4h-databank-prod" --uri="mongodb+srv://ai4h-databank-dev.m0zt2w2.mongodb.net/" --username miladatabank
mongorestore --archive="backup-ai4h-databank-prod-2023-09-29" --nsFrom='ai4h-databank-prod.*' --nsTo='backup-ai4h-databank-prod-2023-09-29.*' --uri="mongodb+srv://ai4h-databank-dev.m0zt2w2.mongodb.net/" --username miladatabank
```

## Logging and Feedback

### Collections

The app supports collecting user feedback and interaction, which is all stored on a cloud instance of mongodb.

The `INSTANCE_TYPE` env. variable determines which database to log to (one of `[prod, dev, or local]`), so that we separate the logs from the different environments.

We record 3 different types of user interactions:

* `interaction`: After a user asks a question and AIR answers it, both the user's question, SAI's answer, cited sources, and other relevant information (parameters, models, etc.) are collected in MongoDB in the `interaction` collection.
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


## Automatic performance evaluation

### Overview

Some evaluations can be run automatically. A list of human validated questions is provided in `data/sample_questions.csv`, and an augmented list of questions is provided in `data/sample_questions_variants.csv`.

The automated performance evaluation will provide some statistics on each category of question. The evaluation is only partial. The current version of the model performs very well on it. As such, it is useful to make sure performance of the system does not degrade when changing the model or features of the system. It is not enough to evaluate improvements. Human evaluation is necessary for this.

#### Relevant

Questions that would be realistically asked by policy makers and lawyers, and whose answer should be in our knowledge base. Both the question and the answer should be relevant.

**Example**: What is the focus of Italy's AI policy strategy?

This category is further divided in 2: original and variants. Originals are questions provided by the OECD. Variants are questions generated by GPT. The goal of the variants is to measure how sensitive the system is to the specific phrasing used. They are studied in more details in the Robustness section below.

#### Irrelevant
Questions that are out of scope. Both the question and the answer should be irrelevant.

**Example**: How can we encourage the use of renewable energy sources and reduce dependence on fossil fuels?

#### Trick
Questions that could realistically be asked, but that the model cannot answer. The question should be marked as relevant, but the answer as irrelevant.

**Example**: Tell me about [made up AI policy].

### Running the evaluation

On GitHub, go to Actions -> Performance -> Run workflow.

Locally, run `pytest performance/test_performance.py --run_expensive`.

Please note that the pipeline makes hundreds of calls to the API and may become costly if used excessively. This is why it has to be triggered manually on GitHub rather than automatically at every commit or merge.


## Experimental features

We have also built out a series of experimental features in the app that are not currently running in production. Here we describe how to activate some of these features. Here are the available features:

- Question reformulation: When a user asks a question, it gets reformulated to be more "optimized" for retrieval using chatGPT
- Number of sources: A user can select how many sources to include in a query to chatGPT (default: 3)
- Documents validation: Given a generated answer, for each source, do an entailment problem to find out which of the sources support the given answer. This is useful to filter out "less relevant" sources but comes at an added cost/latency.
- Finetuning: We support finetuning of the question validator using the OpenAI API.


### Settings tab

For some of these features, it might be useful to allow users to toggle them on/off.
For that purpose, we have also added an optional 'Settings' tab that can be displayed to users.
By default, the settings tab is invisible but can be revealed by setting the `reveal_user_settings = True` variable in the `cfg.py` file.

Currently, the settings tab supports 2 features, namely question reformulation (on/off) and number of sources (a slider).


### Question Reformulation

Question reformulation can be turned on/off by setting the `reformulate_question = False` variable. This will set the default value in the settings tab. It can then be toggled by the user.

To adapt question reformulation (prompts, model, settings, etc.), adapt the following in the `buster_cfg`:


```python
question_reformulator_cfg={
    "completion_kwargs": {
        "model": "gpt-3.5-turbo",
        "stream": False,
        "temperature": 0,
    },
    "system_prompt": """
    Your role is to reformat a user's input into a question that is useful in the context of a semantic retrieval system.
    Reformulate the question in a way that captures the original essence of the question while also adding more relevant details that can be useful in the context of semantic retrieval.""",
},
```

In the app, when the feature is set to on, an additional message will get printed showing the question reformulation. You can edit this message by setting the variables in `cfg.py`:

```python
message_before_reformulation = "I reformulated your answer to: '"
message_after_reformulation = (
    "'\n\nThis is done automatically to increase performance of the tool. You can disable this in the Settings ⚙️ tab."
)
```

### Number of sources

We support passing an arbitrary number of sources in the retrieval process. However, note that the number of sources combined into the prompt must not exceed the allowed token count. Be sure to properly account for the number of tokens and adjust the parameters for generation accordingly. By default, we support 3 sources. This value can be adjusted in the `top_k` setting in the `buster_cfg.retriever_cfg`

### Documents validation

Retrieval always selects `top_k` sources, and while we set a threshold for cosine similarity, it is a value which tends to allow for a lot of false positives. In some circumstances, it might be useful to identify which sources support the generated answer. To do so, we've implemented a validation check where a call is made for each source to ChatGPT to obtain a boolean of relevance. To enable it, set:

```python
validator_cfg={
    "validate_documents": True,  # Validates documents using chatGPT (expensive). Off by default
}
```

Note that this will increase cost and latency to the response.
Also note that while you can enable this feature, currently it will only perform the computation, but not act on it.
To access the result of the computation, you can access it in the `completion.matched_documents['relevant']` column (`matched_documents` is a pandas dataframe and relevant is the column with the computed boolean).
You must then implement logic to decide how to act on this information (e.g. filtering out only relevant sources.)

### Finetuning

Note that it is possible to finetune models using the OpenAI API. A great use-case for this is for the `question_validator`, where we can collect user questions and annotate them as 'relevant', 'not relevant' and run a finetune on that.

We've included helper functions and sample data to help finetune  models on real user data.

This can be found in the `src/finetuning/` folder.

We've also already finetuned a model on the openai platform.

To replace the model for the finetuned one, simply adapt the `completion_kwargs`:

```python
"completion_kwargs": {
    "model": "ft:gpt-3.5-turbo-0613:oecd-ai:first-finetune:8LEyi8pG",
    "stream": False,
    "temperature": 0,
},
```

Note that you need to be a member of the oecd-ai org to have access to the finetuned model.

Evaluating it on the validation set shows the following results:

```
Accuracy using vanilla gpt: 83.67%
Accuracy using finetuned gpt: 89.80%
```
