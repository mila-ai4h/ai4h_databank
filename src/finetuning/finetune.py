import os

from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI(organization=os.environ["OPENAI_ORGANIZATION"])

    training_file = "file-Fqdc3oyjddxQCodwD3wG2Fcx"
    validation_file = "file-vPf2Q42l8hyjeqveOlExjicZ"

    response = client.fine_tuning.jobs.create(
        training_file=training_file,
        model="gpt-3.5-turbo",
        suffix="first-finetune",
        validation_file=validation_file,
    )

    print(response)
