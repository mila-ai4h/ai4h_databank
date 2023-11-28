import os

from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI()

    # First, run the create_dataset.py file and after uploading files, get their file refs on openAI:
    training_file = "file-Fqdc3oyjddxQCodwD3wG2Fcx"
    validation_file = "file-vPf2Q42l8hyjeqveOlExjicZ"

    # Here we can finetune a model
    response = client.fine_tuning.jobs.create(
        training_file=training_file,
        model="gpt-3.5-turbo",
        suffix="first-finetune",
        validation_file=validation_file,
    )

    print(response)
