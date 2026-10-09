import os

from graph import build
from parametres import load_ssm_into_env

SECRETS = ["LLM_API_KEY", "DISCORD_WEBHOOK_URL"]


def lambda_handler(event, context):
    prefix = os.environ.get("SSM_PREFIX")
    if prefix:
        missing = load_ssm_into_env(prefix, SECRETS)
        if missing:
            print(f"Paramètres SSM introuvables : {missing}")
    build().invoke({})
    return {"statusCode": 200, "body": "ok"}