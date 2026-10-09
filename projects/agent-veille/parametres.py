import os


def load_ssm_into_env(prefix: str, names: list[str], client=None) -> list[str]:
    if client is None:
        import boto3

        client = boto3.client("ssm")
    resp = client.get_parameters(Names=[prefix + n for n in names], WithDecryption=True)
    for p in resp["Parameters"]:
        os.environ[p["Name"][len(prefix):]] = p["Value"]
    return [n[len(prefix):] for n in resp.get("InvalidParameters", [])]
