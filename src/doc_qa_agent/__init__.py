"""Document Question and Answer Agent.

A tool-calling agent loop (LangGraph) over a retrieval pipeline (chunk ->
embed -> top-k vector search) running on Amazon Bedrock, with hard limits on
steps/tokens/time, full tool-call tracing, and error feedback to the model.
"""

__version__ = "0.1.0"
