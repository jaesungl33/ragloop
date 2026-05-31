"""Bundled sample corpus and question set for offline evals.

Ten chunks (two per policy document from examples/corpus/) and ten questions
covering every topic. Splitting each file into two chunks makes retrieval
non-trivial so context precision and recall are meaningful even without a
real vector store.
"""
from __future__ import annotations

from ragloop import Document

DOCS: list[Document] = [
    # --- refunds ---
    Document(
        "refunds:0",
        "Northstar Outdoors wants you to shop with confidence. You may request a refund "
        "within 30 days of the delivery date shown on your order confirmation. Refunds apply "
        "to unused products in original packaging with tags attached. Final-sale and "
        "custom-engraved items are not refundable unless they arrive damaged or defective.",
    ),
    Document(
        "refunds:1",
        "To start a return, email support@northstar-outdoors.example with your order number "
        "and reason for the return. We will email a prepaid return label for domestic orders "
        "over $25. Refunds are issued to the original payment method within 5-7 business days "
        "after we receive and inspect the return.",
    ),
    # --- shipping ---
    Document(
        "shipping:0",
        "Northstar Outdoors ships to all 50 U.S. states. We do not ship internationally at "
        "this time.",
    ),
    Document(
        "shipping:1",
        "Standard shipping (3-5 business days) is free on orders over $50. Orders under $50 "
        "incur a flat $6.95 shipping fee. Express shipping (1-2 business days) is available "
        "for $14.95. Orders placed before 2:00 p.m. Eastern Time on business days ship the "
        "same day.",
    ),
    # --- warranty ---
    Document(
        "warranty:0",
        "Most products carry a one-year limited warranty from the date of delivery. Tents "
        "and backpacks include a two-year warranty. Normal wear, misuse, and damage from "
        "accidents are not covered.",
    ),
    Document(
        "warranty:1",
        "To file a warranty claim, contact support with photos of the defect and your order "
        "number. If approved, we will repair, replace, or refund the item. Batteries, "
        "consumables, and items marked 'as-is' in clearance sales are excluded from warranty.",
    ),
    # --- privacy ---
    Document(
        "privacy:0",
        "Northstar Outdoors collects contact details, shipping addresses, payment information, "
        "and browsing activity on our site. We do not sell your personal data to third parties.",
    ),
    Document(
        "privacy:1",
        "You may request access to, correction of, or deletion of your account data by "
        "emailing privacy@northstar-outdoors.example. We respond to verified requests within "
        "30 days as required by applicable law.",
    ),
    # --- support hours ---
    Document(
        "support:0",
        "Live phone and chat support run Monday through Friday, 9:00 a.m. to 6:00 p.m. "
        "Eastern Time. On U.S. federal holidays we are closed; voicemail and email are "
        "monitored the next business day.",
    ),
    Document(
        "support:1",
        "Email support@northstar-outdoors.example any time. We aim to respond within one "
        "business day. Order status, return labels, and warranty claims can be started from "
        "your account dashboard without waiting for an agent.",
    ),
]

QUESTIONS: list[dict] = [
    {
        "question": "What is the refund window?",
        "ground_truth": "You may request a refund within 30 days of the delivery date.",
        "relevant_ids": ["refunds:0"],
    },
    {
        "question": "How do I start a return?",
        "ground_truth": (
            "Email support with your order number and reason for the return "
            "to receive a prepaid return label."
        ),
        "relevant_ids": ["refunds:1"],
    },
    {
        "question": "Is shipping free?",
        "ground_truth": "Standard shipping is free on orders over $50.",
        "relevant_ids": ["shipping:1"],
    },
    {
        "question": "How long does standard shipping take?",
        "ground_truth": "Standard shipping takes 3-5 business days.",
        "relevant_ids": ["shipping:1"],
    },
    {
        "question": "What warranty comes with most products?",
        "ground_truth": "Most products carry a one-year limited warranty against manufacturing defects.",
        "relevant_ids": ["warranty:0"],
    },
    {
        "question": "How do I file a warranty claim?",
        "ground_truth": "Contact support with photos of the defect and your order number.",
        "relevant_ids": ["warranty:1"],
    },
    {
        "question": "Does Northstar sell my personal data?",
        "ground_truth": "No, Northstar does not sell your personal data to third parties.",
        "relevant_ids": ["privacy:0"],
    },
    {
        "question": "When is customer support available?",
        "ground_truth": "Monday through Friday, 9:00 a.m. to 6:00 p.m. Eastern Time.",
        "relevant_ids": ["support:0"],
    },
    {
        "question": "How long until I receive my refund after returning an item?",
        "ground_truth": (
            "Refunds are issued within 5-7 business days after Northstar receives "
            "and inspects the return."
        ),
        "relevant_ids": ["refunds:1"],
    },
    {
        "question": "Do you ship internationally?",
        "ground_truth": "No, Northstar only ships to the 50 U.S. states.",
        "relevant_ids": ["shipping:0"],
    },
]
