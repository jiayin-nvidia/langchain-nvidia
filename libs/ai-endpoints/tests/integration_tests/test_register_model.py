import warnings
from typing import Any

import pytest

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    Model,
    NVIDIAEmbeddings,
    NVIDIARerank,
    register_model,
)


#
# if this test is failing it may be because the function uuids have changed.
# you will have to find the new ones from https://api.nvcf.nvidia.com/v2/nvcf/functions
#
@pytest.mark.parametrize(
    "client, id, endpoint",
    [
        (
            ChatNVIDIA,
            "meta/llama3-8b-instruct",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a5a3ad64-ec2c-4bfc-8ef7-5636f26630fe",
        ),
        (
            NVIDIAEmbeddings,
            "nvidia/nv-embed-v1",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/4c134d37-17f9-4fc6-9f84-1f3a8a03c52c",
        ),
        (
            NVIDIARerank,
            "nv-rerank-qa-mistral-4b:1",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0bf77f50-5c35-4488-8e7a-f49bb1974af6",
        ),
        (
            NVIDIA,
            "bigcode/starcoder2-7b",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/dd7b01e7-732d-4da5-8e8d-315f79165a23",
        ),
    ],
)
def test_registered_model_functional(
    client: type, id: str, endpoint: str, contact_service: Any
) -> None:
    model = Model(id=id, endpoint=endpoint)
    warnings.filterwarnings(
        "ignore", r".*is already registered.*"
    )  # intentionally overridding known models
    warnings.filterwarnings(
        "ignore", r".*Unable to determine validity of.*"
    )  # we aren't passing client & type to Model()
    register_model(model)
    contact_service(client(model=id))
