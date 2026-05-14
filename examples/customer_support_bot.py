"""Realistic customer-support semantic router example with calibration."""

from __future__ import annotations

import hashlib
import os
from typing import Final

import numpy as np

from semantic_router import Route, RouteLayer, ThresholdCalibrator
from semantic_router.encoders import SentenceTransformerEncoder
from semantic_router.encoders.base import BaseEncoder
from semantic_router.utils import normalize

SUPPORT_UTTERANCES: Final[dict[str, list[str]]] = {
    "billing": [
        "why was I charged twice",
        "show my latest invoice",
        "update my payment method",
        "my card was declined",
        "send me a billing receipt",
        "when does my subscription renew",
        "fix my invoice amount",
        "I need a refund for this charge",
        "explain this fee on my account",
        "change my billing address",
    ],
    "technical_support": [
        "the app keeps crashing",
        "I cannot log in on mobile",
        "the website is stuck loading",
        "my dashboard will not refresh",
        "I found a bug in checkout",
        "why is the API returning errors",
        "the page is blank after signing in",
        "help me troubleshoot this problem",
        "the integration stopped syncing",
        "the software froze again",
    ],
    "account": [
        "reset my password",
        "change my email address",
        "delete my account",
        "update my profile settings",
        "turn on two factor authentication",
        "how do I verify my account",
        "I need to reactivate my profile",
        "my username is wrong",
        "log me out of all devices",
        "merge my duplicate accounts",
    ],
    "shipping": [
        "where is my package",
        "track my shipment",
        "my order has not arrived",
        "update the delivery address",
        "when will this item be delivered",
        "the carrier says delayed",
        "can I expedite shipping",
        "my package is marked lost",
        "how do I change delivery instructions",
        "the tracking link is not updating",
    ],
    "returns": [
        "start a return",
        "refund my order",
        "the item arrived damaged",
        "exchange this for another size",
        "what is your return policy",
        "generate a return label",
        "I need to send this item back",
        "my return was not processed",
        "how long does a refund take",
        "cancel the return request",
    ],
    "product_info": [
        "does this come in black",
        "tell me about product features",
        "what size should I buy",
        "is this item waterproof",
        "compare these two plans",
        "do you have more technical specs",
        "what materials is this made from",
        "is the premium plan worth it",
        "show me compatibility details",
        "does this support team billing",
    ],
    "escalation": [
        "I want to speak to a manager",
        "this issue is urgent",
        "please escalate my case",
        "I already contacted support twice",
        "I need a supervisor to review this",
        "this is unacceptable",
        "can someone senior take over",
        "open a priority ticket for me",
        "I need executive support now",
        "this complaint needs escalation",
    ],
    "smalltalk": [
        "hello there",
        "good morning",
        "thanks for your help",
        "how are you doing",
        "you have been very helpful",
        "have a nice day",
        "that is all for now",
        "just saying hi",
        "appreciate your support",
        "nice chatting with you",
    ],
}

SUPPORT_KEYWORDS: Final[dict[str, set[str]]] = {
    "billing": {
        "charge",
        "invoice",
        "payment",
        "card",
        "receipt",
        "billing",
        "fee",
        "refund",
    },
    "technical_support": {
        "crash",
        "bug",
        "error",
        "loading",
        "froze",
        "sync",
        "troubleshoot",
        "login",
    },
    "account": {
        "password",
        "email",
        "account",
        "profile",
        "username",
        "verify",
        "reactivate",
    },
    "shipping": {
        "package",
        "shipment",
        "delivery",
        "carrier",
        "tracking",
        "arrived",
        "ship",
    },
    "returns": {
        "return",
        "refund",
        "exchange",
        "damaged",
        "label",
        "policy",
        "send back",
    },
    "product_info": {
        "feature",
        "size",
        "waterproof",
        "compare",
        "spec",
        "materials",
        "compatibility",
    },
    "escalation": {
        "manager",
        "urgent",
        "escalate",
        "supervisor",
        "priority",
        "complaint",
        "senior",
    },
    "smalltalk": {"hello", "morning", "thanks", "hi", "nice", "appreciate", "day"},
}


class SupportFallbackEncoder(BaseEncoder):
    """Keyword-aware fallback encoder for the customer support demo."""

    def __init__(self) -> None:
        super().__init__()
        self._route_names = list(SUPPORT_KEYWORDS)
        self._anchors = {
            name: normalize(np.eye(len(self._route_names), dtype=np.float32)[index])
            for index, name in enumerate(self._route_names)
        }

    def _category(self, text: str) -> str:
        lowered = text.lower()
        for route_name, keywords in SUPPORT_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return route_name
        return "smalltalk"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 0.025, size=len(self._route_names)).astype(
                np.float32
            )
            vectors.append(normalize(self._anchors[self._category(text)] + noise))
        return np.stack(vectors).astype(np.float32, copy=False)

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return len(self._route_names)

    @property
    def name(self) -> str:
        return "support-fallback"


def build_encoder() -> BaseEncoder:
    if os.getenv("SEMANTIC_ROUTER_DOWNLOAD_MODELS") == "1":
        return SentenceTransformerEncoder(
            model_name="all-MiniLM-L6-v2",
            show_progress=False,
        )
    return SupportFallbackEncoder()


def build_layer() -> RouteLayer:
    routes = [
        Route(
            name=name,
            utterances=utterances,
            description=f"Customer support route: {name}",
        )
        for name, utterances in SUPPORT_UTTERANCES.items()
    ]
    return RouteLayer(routes=routes, encoder=build_encoder())


def print_results(layer: RouteLayer) -> None:
    test_queries = [
        ("my invoice looks wrong this month", "billing"),
        ("the mobile app crashes on launch", "technical_support"),
        ("please change my login email", "account"),
        ("where is my package right now", "shipping"),
        ("I want to start a return for this item", "returns"),
        ("does this jacket come in blue", "product_info"),
        ("this issue is urgent and needs a manager", "escalation"),
        ("thanks so much for helping me", "smalltalk"),
        ("my card payment failed again", "billing"),
        ("the API integration stopped syncing orders", "technical_support"),
        ("log me out of every device", "account"),
        ("the carrier says my shipment is delayed", "shipping"),
        ("how long does a refund usually take", "returns"),
        ("can you compare the premium and team plans", "product_info"),
        ("please escalate my complaint immediately", "escalation"),
    ]
    header = f"{'Query':<52} {'Expected':<20} {'Matched':<20} {'Score':>6}"
    print(header)
    print("-" * len(header))
    for query, expected in test_queries:
        match = layer.route(query)
        matched = "None" if match is None else match.name
        score = "-" if match is None else f"{match.score:.3f}"
        print(f"{query[:52]:<52} {expected:<20} {matched:<20} {score:>6}")

    calibrator = ThresholdCalibrator(layer)
    calibration_dataset = test_queries + [
        ("hello friend", "smalltalk"),
        ("nonsense flarn gibberish", None),
        ("my package never moved from the depot", "shipping"),
    ]
    result = calibrator.calibrate(
        calibration_dataset,
        thresholds=np.linspace(0.50, 0.95, 10).tolist(),
        metric="f1",
    )
    print("\nBest global threshold:", f"{result.best_global_threshold:.3f}")
    print(result.summary())


def main() -> None:
    layer = build_layer()
    print(
        f"Loaded {len(layer.list_routes())} support routes with encoder: "
        f"{layer.encoder.name}"
    )
    print_results(layer)


if __name__ == "__main__":
    main()
