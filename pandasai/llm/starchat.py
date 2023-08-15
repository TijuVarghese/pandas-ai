""" StarChat LLM
This module is to run the StarChat HF API hosted and maintained by HuggingFace.co.
To generate HF_TOKEN go to https://huggingface.co/settings/tokens after creating Account
on the platform.

Example:
    Use below example to call Falcon Model

from pandasai.llm.starchat import StarChat
"""


from ..helpers import load_dotenv
from .base import HuggingFaceLLM

load_dotenv()


class StarChat(HuggingFaceLLM):

    """StarChat LLM API

    A base HuggingFaceLLM class is extended to use StarChat model.

    """

    api_token: str
    _api_url: str = (
        "https://api-inference.huggingface.co/models/HuggingFaceH4/starchat-beta"
    )
    _max_retries: int = 5

    @property
    def type(self) -> str:
        return "starchat"
