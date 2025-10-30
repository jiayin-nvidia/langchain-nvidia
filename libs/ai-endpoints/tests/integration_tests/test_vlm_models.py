import base64
import os
from typing import Any, Dict, List, Union

import pytest
import requests
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

# todo: multiple texts
# todo: accuracy tests

#
# API Specification -
#
#  - User message may contain 1 or more image_url
#  - image_url contains a url and optional detail
#  - detail is one of "low", "high" or "auto" (default)
#  - url is either a url to an image or base64 encoded image
#  - format for base64 is "data:image/png;{type}},..."
#  - supported image types are png, jpeg (or jpg), webp, gif (non-animated)
#

#
# note: differences between api catalog and openai api
#  - openai api supports server-side image download, api catalog does not consistently
#   - ChatNVIDIA does client side download to simulate the same behavior
#  - ChatNVIDIA will automatically read local files and convert them to base64
#  - openai api always uses {"image_url": {"url": "..."}}
#     where api catalog sometimes uses {"image_url": "..."}
#


@pytest.mark.parametrize(
    "content",
    [
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Boston_-_panoramio_(23).jpg/2560px-Boston_-_panoramio_(23).jpg"
                },
            }
        ],
        [{"type": "image_url", "image_url": {"url": "tests/data/nvidia-picasso.jpg"}}],
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"""data:image/jpg;base64,{
                        base64.b64encode(
                            open('tests/data/nvidia-picasso.jpg', 'rb').read()
                        ).decode('utf-8')
                    }"""
                },
            }
        ],
        f"""<img src="data:image/jpg;base64,{
            base64.b64encode(
                open('tests/data/nvidia-picasso.jpg', 'rb').read()
            ).decode('utf-8')
        }"/>""",
    ],
    ids=["url", "file", "data", "tag"],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "stream", "ainvoke", "astream"],
)
async def test_vlm_input_style(
    vlm_model: str,
    mode: dict,
    func: str,
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    if func == "invoke":
        response = chat.invoke([HumanMessage(content=content)])
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
    elif func == "stream":
        for token in chat.stream([HumanMessage(content=content)]):
            assert isinstance(token.content, str)
    elif func == "ainvoke":
        response = await chat.ainvoke([HumanMessage(content=content)])
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
    elif func == "astream":
        async for token in chat.astream([HumanMessage(content=content)]):
            assert isinstance(token.content, str)


@pytest.mark.parametrize(
    "detail",
    ["low", "high", "auto"],
    ids=["low", "high", "auto"],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_vlm_detail_accepted(
    vlm_model: str,
    mode: dict,
    detail: str,
    func: str,
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tests/data/nvidia-picasso.jpg",
                        "detail": detail,
                    },
                }
            ]
        )
    ]

    if func == "invoke":
        response = chat.invoke(messages)
    else:  # ainvoke
        response = await chat.ainvoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
    # assert "cat" in response.content.lower()


@pytest.mark.parametrize(
    "invalid_detail",
    [None, "None", "medium", "HIGH", ""],
    ids=["None", "None-str", "medium", "HIGH", ""],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_vlm_detail_invalid(
    vlm_model: str,
    mode: dict,
    func: str,
    invalid_detail: str,
) -> None:
    """Test that invalid detail values raise ValueError."""
    chat = ChatNVIDIA(model=vlm_model, **mode)

    message = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tests/data/nvidia-picasso.jpg",
                        "detail": invalid_detail,
                    },
                }
            ]
        )
    ]

    with pytest.raises(ValueError, match="Invalid detail value"):
        if func == "invoke":
            chat.invoke(message)
        else:
            await chat.ainvoke(message)


@pytest.mark.parametrize(
    "img",
    [
        "tests/data/nvidia-picasso.jpg",
        "tests/data/nvidia-picasso.png",
        "tests/data/nvidia-picasso.webp",
        "tests/data/nvidia-picasso.gif",
    ],
    ids=["jpg", "png", "webp", "gif"],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_vlm_image_type(
    vlm_model: str,
    mode: dict,
    img: str,
    func: str,
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img,
                    },
                }
            ]
        )
    ]

    if func == "invoke":
        response = chat.invoke(messages)
    else:  # ainvoke
        response = await chat.ainvoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_vlm_image_large(
    vlm_model: str,
    mode: dict,
    func: str,
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tests/data/nvidia-picasso-large.png",
                    },
                }
            ]
        )
    ]

    if func == "invoke":
        response = chat.invoke(messages)
    else:  # ainvoke
        response = await chat.ainvoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_vlm_no_images(
    vlm_model: str,
    mode: dict,
    func: str,
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    messages = [HumanMessage(content="What is the capital of Massachusetts?")]

    if func == "invoke":
        response = chat.invoke(messages)
    else:  # ainvoke
        response = await chat.ainvoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


# mark this test as xfail
@pytest.mark.xfail(
    reason="Test fails when using meta/llama-3.2-11b-vision-instruct model"
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_vlm_two_images(
    vlm_model: str,
    mode: dict,
    func: str,
) -> None:
    chat = ChatNVIDIA(model=vlm_model, **mode)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tests/data/nvidia-picasso.jpg",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "tests/data/nvidia-picasso.jpg",
                    },
                },
            ]
        )
    ]

    if func == "invoke":
        response = chat.invoke(messages)
    else:  # ainvoke
        response = await chat.ainvoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.fixture(scope="session")
def asset_id() -> str:
    # create an asset following -
    #  https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/assets.html

    def create_asset_and_get_upload_url(
        token: str, content_type: str, description: str
    ) -> dict:
        url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        data = {"contentType": content_type, "description": description}
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def upload_image_to_presigned_url(
        image_path: str, upload_url: str, content_type: str, description: str
    ) -> None:
        headers = {
            "Content-Type": content_type,
            "x-amz-meta-nvcf-asset-description": description,
        }
        with open(image_path, "rb") as image_file:
            response = requests.put(upload_url, headers=headers, data=image_file)
            response.raise_for_status()

    content_type = "image/jpg"
    description = "lc-nv-ai-e-test-nvidia-picasso"

    asset_info = create_asset_and_get_upload_url(
        os.environ["NVIDIA_API_KEY"], content_type, description
    )
    asset_id = asset_info["assetId"]

    upload_image_to_presigned_url(
        "tests/data/nvidia-picasso.jpg",
        asset_info["uploadUrl"],
        content_type,
        description,
    )

    return asset_id


@pytest.mark.parametrize(
    "content",
    [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpg;asset_id,{asset_id}"},
                    }
                ],
            }
        ],
        [
            """<img src="data:image/jpg;asset_id,{asset_id}"/>""",
        ],
        """<img src="data:image/jpg;asset_id,{asset_id}"/>""",
    ],
    ids=["data", "list-of-tag", "tag"],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "stream", "ainvoke", "astream"],
)
async def test_vlm_asset_id(
    vlm_model: str,
    mode: dict,
    content: Union[str, List[Union[str, Dict[str, Any]]]],
    func: str,
    asset_id: str,
) -> None:
    def fill(
        item: Any,
        asset_id: str,
    ) -> Union[str, Any]:
        # do not mutate item, mutation will cause cross test contamination
        result: Any
        if isinstance(item, str):
            result = item.format(asset_id=asset_id)
        elif isinstance(item, BaseMessage):
            result = item.model_copy(update={"content": fill(item.content, asset_id)})
        elif isinstance(item, list):
            result = [fill(sub_item, asset_id) for sub_item in item]
        elif isinstance(item, dict):
            result = {key: fill(value, asset_id) for key, value in item.items()}
        return result

    content = fill(content, asset_id)

    chat = ChatNVIDIA(model=vlm_model, **mode)
    if func == "invoke":
        response = chat.invoke(content)
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
    elif func == "stream":
        for token in chat.stream(content):
            assert isinstance(token.content, str)
    elif func == "ainvoke":
        response = await chat.ainvoke(content)
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
    elif func == "astream":
        async for token in chat.astream(content):
            assert isinstance(token.content, str)
