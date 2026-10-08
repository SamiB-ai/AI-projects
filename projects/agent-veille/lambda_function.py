from graph import build


def lambda_handler(event, context):
    build().invoke({})
    return {"statusCode": 200, "body": "ok"}
