import json
import os


class LocalSeenStore:
    def __init__(self, path: str = "seen.json"):
        self.path = path

    def load(self) -> set[str]:
        if not os.path.exists(self.path):
            return set()
        with open(self.path, encoding="utf-8") as f:
            return set(json.load(f))

    def save(self, ids: set[str]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(sorted(ids), f)


class S3SeenStore:
    def __init__(self, bucket: str, key: str = "seen.json", client=None):
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.client = client
        self.bucket = bucket
        self.key = key

    def load(self) -> set[str]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if code == "NoSuchKey":
                return set()
            raise
        return set(json.loads(obj["Body"].read()))

    def save(self, ids: set[str]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key,
            Body=json.dumps(sorted(ids)).encode(),
            ContentType="application/json",
        )


def get_store():
    bucket = os.environ.get("S3_BUCKET")
    return S3SeenStore(bucket) if bucket else LocalSeenStore()