"""A deliberately harder eval scenario to stress the self-correction loop.

The standard scenario (corpus.py) is small and clean, so a one-shot baseline
already does well. This scenario adds:

1. **Distractor chunks** — topically adjacent passages that do NOT answer certain
   questions (discounts, accepted cards, warehouse location). A one-shot baseline
   tends to retrieve a distractor and answer from it; the loop's critic should
   notice the answer isn't grounded in a chunk that actually addresses the
   question and decline.
2. **Distractor-trap questions** — unanswerable questions that *look* answerable
   because a tempting distractor exists (military discount, financing, gift
   wrapping...). The corpus genuinely does not answer them.
3. **Retrieval-hard questions** — answerable, but worded far from the source
   chunk, so a single semantic search may miss and the planner's reformulation /
   retry has to recover the gold chunk.

Everything is still grounded in the Northstar Outdoors corpus; only the
difficulty changes.
"""
from __future__ import annotations

from ragloop import Document

from .corpus import DOCS as _BASE_DOCS

# Topically adjacent passages that do NOT answer the trap questions below.
_DISTRACTORS: list[Document] = [
    Document(
        "promotions:0",
        "Northstar Outdoors runs seasonal sales throughout the year. Sign up for "
        "our email newsletter to be the first to hear about promotions and limited "
        "time offers on outdoor gear.",
    ),
    Document(
        "payments:0",
        "At checkout we accept Visa, Mastercard, American Express, Discover, and "
        "PayPal. Payment is charged when your order ships.",
    ),
    Document(
        "company:0",
        "Northstar Outdoors was founded in 2014 and ships from our warehouse in "
        "Portland, Maine. Our gear is designed for hikers, campers, and climbers.",
    ),
    Document(
        "account:0",
        "Create a Northstar account to track orders, save addresses, and start "
        "returns or warranty claims from your dashboard.",
    ),
]

HARD_DOCS: list[Document] = list(_BASE_DOCS) + _DISTRACTORS

HARD_QUESTIONS: list[dict] = [
    # --- retrieval-hard but answerable (vocabulary gap / needs reasoning) ---
    {
        "question": "If my order total is only $40, what will I pay to have it delivered?",
        "ground_truth": "Orders under $50 pay a flat $6.95 standard shipping fee.",
        "relevant_ids": ["shipping:1"],
    },
    {
        "question": "My tent ripped at the seam after 14 months — is that covered?",
        "ground_truth": "Yes; tents carry a two-year warranty against manufacturing defects.",
        "relevant_ids": ["warranty:0"],
    },
    {
        "question": "How soon will money come back to my card after I send an item back?",
        "ground_truth": "Refunds are issued within 5-7 business days after the return is received and inspected.",
        "relevant_ids": ["refunds:1"],
    },
    {
        "question": "Can someone in Toronto order from you?",
        "ground_truth": "No; Northstar ships only to the 50 U.S. states, not internationally.",
        "relevant_ids": ["shipping:0"],
    },
    {
        "question": "What hours can I reach a person by phone?",
        "ground_truth": "Phone support runs Monday-Friday, 9:00 a.m. to 6:00 p.m. Eastern Time.",
        "relevant_ids": ["support:0"],
    },
    {
        "question": "Which credit cards can I use to pay?",
        "ground_truth": "Visa, Mastercard, American Express, Discover, and PayPal.",
        "relevant_ids": ["payments:0"],
    },
    {
        "question": "Which city do orders ship out of?",
        "ground_truth": "Orders ship from Northstar's warehouse in Portland, Maine.",
        "relevant_ids": ["company:0"],
    },
    {
        "question": "By what time do I need to order to get it sent the same day?",
        "ground_truth": "Orders placed before 2:00 p.m. Eastern Time on business days ship the same day.",
        "relevant_ids": ["shipping:1"],
    },
    {
        "question": "I had my hiking knife engraved with my name — can I return it?",
        "ground_truth": "No; custom-engraved items are not refundable unless they arrive damaged or defective.",
        "relevant_ids": ["refunds:0"],
    },
    {
        "question": "What do I need to send in to start a warranty claim?",
        "ground_truth": "Contact support with photos of the defect and your order number.",
        "relevant_ids": ["warranty:1"],
    },
    # --- distractor traps: unanswerable, but a tempting chunk exists ---
    {
        "question": "Do you offer a military or veteran discount?",
        "ground_truth": "The provided sources do not mention any military or veteran discount.",
        "relevant_ids": [],
        "answerable": False,
    },
    {
        "question": "Can I pay in monthly installments or use buy-now-pay-later?",
        "ground_truth": "The provided sources list accepted cards but do not mention financing or installment plans.",
        "relevant_ids": [],
        "answerable": False,
    },
    {
        "question": "Do you offer gift wrapping?",
        "ground_truth": "The provided sources do not mention gift wrapping.",
        "relevant_ids": [],
        "answerable": False,
    },
    {
        "question": "Is there a loyalty or rewards program?",
        "ground_truth": "The provided sources do not mention a loyalty or rewards program.",
        "relevant_ids": [],
        "answerable": False,
    },
    {
        "question": "Who is the CEO of Northstar Outdoors?",
        "ground_truth": "The provided sources do not name the CEO or any individual staff.",
        "relevant_ids": [],
        "answerable": False,
    },
    {
        "question": "Do you price match other retailers?",
        "ground_truth": "The provided sources do not mention price matching.",
        "relevant_ids": [],
        "answerable": False,
    },
]
