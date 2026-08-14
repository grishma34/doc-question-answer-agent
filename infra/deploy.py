"""Deploy the hosted demo: S3 + CloudFront frontend, Lambda backend.

    PYTHONPATH=src python infra/deploy.py

Creates (all in us-east-1, all free or near-free at rest):
  - DynamoDB table  docqa-demo-usage        (daily request counter)
  - IAM role        docqa-demo-lambda-role  (logs + bedrock + counter)
  - Lambda          docqa-demo-api          (bundled index, no 3p deps)
  - HTTP API        docqa-demo-api          (API Gateway proxy to Lambda;
                     the handler requires a secret header that only
                     CloudFront injects, so direct calls get 403)
  - S3 bucket       docqa-demo-<account>    (private; OAC access only)
  - CloudFront distribution:  /  -> S3,  /api/* -> API Gateway
  - AWS Budget      $5/month with email alert

Note: Lambda function URLs (both with CloudFront OAC/IAM auth and with
open auth) were rejected at the service level on this account, so the
API sits behind an API Gateway HTTP API instead (~$1/M requests).

Rerunnable: existing resources are reused, code/site are re-uploaded.
"""

import io
import json
import secrets
import time
import zipfile
from pathlib import Path

import boto3
import numpy as np

REGION = "us-east-1"
APP = "docqa-demo"
BUDGET_EMAIL = "akattela1@gmail.com"
INFRA = Path(__file__).parent
REPO = INFRA.parent

session = boto3.Session(region_name=REGION)
sts = session.client("sts")
ACCOUNT = sts.get_caller_identity()["Account"]
BUCKET = f"{APP}-{ACCOUNT}"


def build_index_json() -> bytes:
    """Convert the local numpy index into a plain-JSON bundle for Lambda."""
    vectors = np.load(REPO / "index" / "vectors.npy")
    chunks = json.loads((REPO / "index" / "chunks.json").read_text())
    entries = [
        {"chunk_id": c["chunk_id"], "text": c["text"],
         "vector": [round(float(x), 6) for x in vectors[i]]}
        for i, c in enumerate(chunks)
    ]
    return json.dumps(entries).encode()


def build_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lambda_handler.py", (INFRA / "lambda_handler.py").read_text())
        z.writestr("index.json", build_index_json())
    return buf.getvalue()


def ensure_table(ddb):
    try:
        ddb.create_table(
            TableName=f"{APP}-usage",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print("dynamodb: created")
    except ddb.exceptions.ResourceInUseException:
        print("dynamodb: exists")
    ddb.get_waiter("table_exists").wait(TableName=f"{APP}-usage")


def ensure_role(iam) -> str:
    role_name = f"{APP}-lambda-role"
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow",
                       "Principal": {"Service": "lambda.amazonaws.com"},
                       "Action": "sts:AssumeRole"}],
    }
    try:
        role = iam.create_role(RoleName=role_name,
                               AssumeRolePolicyDocument=json.dumps(trust))
        print("iam role: created")
        created = True
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=role_name)
        print("iam role: exists")
        created = False
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow",
             "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
             "Resource": "arn:aws:logs:*:*:*"},
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["dynamodb:UpdateItem"],
             "Resource": f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{APP}-usage"},
        ],
    }
    iam.put_role_policy(RoleName=role_name, PolicyName=f"{APP}-policy",
                        PolicyDocument=json.dumps(policy))
    if created:
        time.sleep(12)  # allow role to propagate before Lambda uses it
    return role["Role"]["Arn"]


def ensure_lambda(lam, role_arn: str) -> tuple[str, str]:
    name = f"{APP}-api"
    code = build_zip()

    # keep an existing origin secret on redeploys so CloudFront stays in sync
    origin_secret = None
    try:
        current = lam.get_function_configuration(FunctionName=name)
        origin_secret = current.get("Environment", {}).get("Variables", {}).get("ORIGIN_SECRET")
    except lam.exceptions.ResourceNotFoundException:
        pass
    origin_secret = origin_secret or secrets.token_hex(24)

    env = {"Variables": {
        "BEDROCK_REGION": REGION,
        "USAGE_TABLE": f"{APP}-usage",
        "DAILY_REQUEST_LIMIT": "40",
        "ORIGIN_SECRET": origin_secret,
    }}
    try:
        lam.create_function(
            FunctionName=name, Runtime="python3.12", Handler="lambda_handler.handler",
            Role=role_arn, Code={"ZipFile": code}, Timeout=30, MemorySize=256,
            Environment=env,
        )
        print("lambda: created")
    except lam.exceptions.ResourceConflictException:
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        lam.update_function_code(FunctionName=name, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        lam.update_function_configuration(FunctionName=name, Environment=env)
        print("lambda: updated")
    lam.get_waiter("function_active_v2").wait(FunctionName=name)

    return origin_secret


def ensure_http_api(apigw, lam) -> str:
    """API Gateway HTTP API proxying every route to the Lambda."""
    name = f"{APP}-api"
    fn_arn = lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]
    existing = [a for a in apigw.get_apis()["Items"] if a["Name"] == name]
    if existing:
        api = existing[0]
        print("http api: exists")
    else:
        api = apigw.create_api(Name=name, ProtocolType="HTTP", Target=fn_arn)
        print("http api: created")
    try:
        lam.add_permission(
            FunctionName=name, StatementId="apigw-invoke",
            Action="lambda:InvokeFunction", Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{REGION}:{ACCOUNT}:{api['ApiId']}/*/*",
        )
    except lam.exceptions.ResourceConflictException:
        pass
    return api["ApiEndpoint"]


def ensure_bucket(s3):
    try:
        s3.create_bucket(Bucket=BUCKET)  # us-east-1 needs no LocationConstraint
        print("s3 bucket: created")
    except (s3.exceptions.BucketAlreadyOwnedByYou, s3.exceptions.BucketAlreadyExists):
        print("s3 bucket: exists")
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": False, "RestrictPublicBuckets": False,
        },
    )
    s3.put_object(
        Bucket=BUCKET, Key="index.html",
        Body=(INFRA / "frontend" / "index.html").read_bytes(),
        ContentType="text/html; charset=utf-8", CacheControl="max-age=300",
    )
    print("s3: index.html uploaded")


def ensure_oac(cf, name: str, origin_type: str) -> str:
    existing = cf.list_origin_access_controls().get("OriginAccessControlList", {})
    for item in existing.get("Items", []):
        if item["Name"] == name:
            return item["Id"]
    resp = cf.create_origin_access_control(OriginAccessControlConfig={
        "Name": name, "SigningProtocol": "sigv4", "SigningBehavior": "always",
        "OriginAccessControlOriginType": origin_type,
    })
    return resp["OriginAccessControl"]["Id"]


def ensure_query_only_policy(cf) -> str:
    """Origin request policy forwarding query strings only."""
    for i in cf.list_origin_request_policies(Type="custom")["OriginRequestPolicyList"].get("Items", []):
        if i["OriginRequestPolicy"]["OriginRequestPolicyConfig"]["Name"] == f"{APP}-query-only":
            return i["OriginRequestPolicy"]["Id"]
    resp = cf.create_origin_request_policy(OriginRequestPolicyConfig={
        "Name": f"{APP}-query-only",
        "Comment": "Forward query strings only",
        "HeadersConfig": {"HeaderBehavior": "none"},
        "CookiesConfig": {"CookieBehavior": "none"},
        "QueryStringsConfig": {"QueryStringBehavior": "all"},
    })
    return resp["OriginRequestPolicy"]["Id"]


def ensure_distribution(cf, api_endpoint: str, s3_oac: str, origin_secret: str) -> tuple[str, str]:
    lambda_host = api_endpoint.replace("https://", "").strip("/")
    s3_domain = f"{BUCKET}.s3.{REGION}.amazonaws.com"
    orp_id = ensure_query_only_policy(cf)

    for d in cf.list_distributions().get("DistributionList", {}).get("Items", []):
        if d.get("Comment") == APP:
            print("cloudfront: exists")
            return d["Id"], d["DomainName"]

    config = {
        "CallerReference": f"{APP}-{int(time.time())}",
        "Comment": APP,
        "Enabled": True,
        "DefaultRootObject": "index.html",
        "PriceClass": "PriceClass_100",
        "Origins": {"Quantity": 2, "Items": [
            {
                "Id": "s3-site", "DomainName": s3_domain,
                "OriginAccessControlId": s3_oac,
                "S3OriginConfig": {"OriginAccessIdentity": ""},
            },
            {
                "Id": "lambda-api", "DomainName": lambda_host,
                "CustomHeaders": {"Quantity": 1, "Items": [
                    {"HeaderName": "x-origin-verify", "HeaderValue": origin_secret},
                ]},
                "CustomOriginConfig": {
                    "HTTPPort": 80, "HTTPSPort": 443,
                    "OriginProtocolPolicy": "https-only",
                    "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
                },
            },
        ]},
        "DefaultCacheBehavior": {
            "TargetOriginId": "s3-site",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"],
                               "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
            "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",  # CachingOptimized
            "Compress": True,
        },
        "CacheBehaviors": {"Quantity": 1, "Items": [{
            "PathPattern": "/api/*",
            "TargetOriginId": "lambda-api",
            "ViewerProtocolPolicy": "https-only",
            "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"],
                               "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
            "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",  # CachingDisabled
            "OriginRequestPolicyId": orp_id,
            "Compress": False,
        }]},
    }
    resp = cf.create_distribution(DistributionConfig=config)
    print("cloudfront: created")
    return resp["Distribution"]["Id"], resp["Distribution"]["DomainName"]


def wire_permissions(s3, dist_id: str):
    dist_arn = f"arn:aws:cloudfront::{ACCOUNT}:distribution/{dist_id}"
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowCloudFrontOAC", "Effect": "Allow",
            "Principal": {"Service": "cloudfront.amazonaws.com"},
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{BUCKET}/*",
            "Condition": {"StringEquals": {"AWS:SourceArn": dist_arn}},
        }],
    }
    s3.put_bucket_policy(Bucket=BUCKET, Policy=json.dumps(bucket_policy))
    print("s3: bucket policy set (CloudFront-only)")


def ensure_budget():
    budgets = session.client("budgets", region_name="us-east-1")
    try:
        budgets.create_budget(
            AccountId=ACCOUNT,
            Budget={
                "BudgetName": f"{APP}-monthly-cap",
                "BudgetLimit": {"Amount": "5", "Unit": "USD"},
                "TimeUnit": "MONTHLY", "BudgetType": "COST",
            },
            NotificationsWithSubscribers=[{
                "Notification": {"NotificationType": "ACTUAL",
                                 "ComparisonOperator": "GREATER_THAN",
                                 "Threshold": 80.0, "ThresholdType": "PERCENTAGE"},
                "Subscribers": [{"SubscriptionType": "EMAIL", "Address": BUDGET_EMAIL}],
            }],
        )
        print(f"budget: $5/month alert -> {BUDGET_EMAIL}")
    except budgets.exceptions.DuplicateRecordException:
        print("budget: exists")
    except Exception as e:
        print(f"budget: skipped ({type(e).__name__}: {e})")


def main():
    ddb = session.client("dynamodb")
    iam = session.client("iam")
    lam = session.client("lambda")
    s3 = session.client("s3")
    cf = session.client("cloudfront")

    apigw = session.client("apigatewayv2")

    ensure_table(ddb)
    role_arn = ensure_role(iam)
    origin_secret = ensure_lambda(lam, role_arn)
    api_endpoint = ensure_http_api(apigw, lam)
    print("api endpoint:", api_endpoint)
    ensure_bucket(s3)
    s3_oac = ensure_oac(cf, f"{APP}-s3-oac", "s3")
    dist_id, domain = ensure_distribution(cf, api_endpoint, s3_oac, origin_secret)
    wire_permissions(s3, dist_id)
    ensure_budget()

    print(f"\nDistribution: {dist_id}")
    print(f"URL: https://{domain}")
    print("(CloudFront deployment takes ~5-15 minutes on first creation)")


if __name__ == "__main__":
    main()
