"""Embeddings via Amazon Bedrock (Titan Text Embeddings V2)."""

import json

import boto3
import numpy as np


class BedrockEmbedder:
    """Embeds text with Titan V2 through the bedrock-runtime API.

    Titan V2 pricing is $0.00002 per 1K input tokens, so embedding an entire
    small document set costs a fraction of a cent.
    """

    def __init__(self, model_id: str, region: str, dimensions: int = 512, client=None):
        self.model_id = model_id
        self.dimensions = dimensions
        self._client = client or boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text: str) -> np.ndarray:
        body = json.dumps(
            {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        )
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return np.asarray(payload["embedding"], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])
