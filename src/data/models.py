from __future__ import annotations

from typing import Optional, Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Deal(BaseModel):
    """
    A normalized deal surfaced by the Scanner.
    """

    product_description: str = Field(
        description=(
            "Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more "
            "important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. "
            "There should be a short paragraph of text for each item you choose."
        )
    )
    price: float = Field(
        description=(
            "The actual price of this product, as advertised in the deal. Be sure to give the actual price; "
            'for example, if a deal is described as "$100 off the usual $300 price", you should respond with $200.'
        )
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    """
    A list of Deals returned by the Scanner LLM.
    """

    deals: list[Deal] = Field(
        description=(
            "Your selection of the 5 deals that have the most detailed, high quality description and the most clear "
            "price. You should be confident that the price reflects the deal, that it is a good deal, with a clear "
            "description"
        )
    )


class Opportunity(BaseModel):
    """
    A potential bargain: deal_price vs estimated_true_value.
    """

    deal: Deal
    estimate: float
    discount: float


class Item(BaseModel):
    """
    A product datapoint (title/category/price/text) used for training and RAG vector DB ingestion.
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct Items"""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )

