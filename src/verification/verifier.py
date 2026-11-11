"""
Selective verification (report §14).

`verification_triggers` decides *whether* to verify (cost control);
`AnswerVerifier.verify` performs the check and reports the Unsupported Claim
Rate (report §22.2); `needs_revision` decides whether to spend a second model
call revising the answer.
"""

import re
from dataclasses import dataclass, field

from src.verification.claims import EvidenceMapper, extract_claims

RISKY_CONTENT = re.compile(
    r"\b(\d{4}|\d+(\.\d+)?\s*(%|percent|ms|gb|mb|hz|v|w|kg|m/s))\b|"
    r"\b(invented|introduced|proposed|discovered|founded|first)\b",
    re.IGNORECASE,
)


@dataclass
class VerificationReport:
    performed: bool
    triggers: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    total_claims: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    unsupported_claim_rate: float | None = None
    invalid_citations: int = 0
    revised: bool = False
    note: str = ""

    def to_dict(self):
        data = dict(self.__dict__)
        data["claims"] = [c.to_dict() for c in self.claims]
        return data


def verification_triggers(analysis, retrieval_signal=None, answer=None, answer_mode=None):
    """Return the list of reasons to verify (empty list = skip verification)."""
    triggers = []
    if retrieval_signal is not None and retrieval_signal.get("strength") == "weak":
        triggers.append("retrieved evidence is weakly relevant")
    # `conflict_risk` (near-tied top articles) is reported in the routing log but
    # is not a trigger by itself: related articles routinely tie (e.g. "Neural
    # network" and "Backpropagation"), which is not evidence of disagreement.
    # A content-level contradiction check would be needed for that trigger.
    if analysis is not None:
        if analysis.intent in ("numerical", "procedure"):
            triggers.append(f"{analysis.intent} answer depends on exact steps/values")
        if analysis.ambiguous:
            triggers.append("question is ambiguous")
        if analysis.complexity >= 3:
            triggers.append("multi-part / complex question")
    if answer_mode == "10-mark":
        triggers.append("long answer with many claims")
    if answer and RISKY_CONTENT.search(answer):
        triggers.append("answer contains dates, figures or historical claims")
    return triggers


class AnswerVerifier:

    def __init__(self, embedding_model, entailment_fn=None, evidence_cache=None):
        self.mapper = EvidenceMapper(embedding_model, entailment_fn=entailment_fn,
                                     evidence_cache=evidence_cache)

    def verify(self, answer, passages, triggers=None, question=None):
        claims = extract_claims(answer)
        report = VerificationReport(performed=True, triggers=list(triggers or []))
        if not passages:
            report.note = "No evidence passages available; claims could not be checked."
            report.claims = claims
            report.total_claims = len(claims)
            return report

        from src.verification.claims import numbers_in
        self.mapper.map_claims(claims, passages, known_numbers=numbers_in(question or ""))
        report.claims = claims
        report.total_claims = len(claims)
        report.supported = sum(c.label == "supported" for c in claims)
        report.partial = sum(c.label == "partial" for c in claims)
        report.unsupported = sum(c.label == "unsupported" for c in claims)
        report.invalid_citations = sum(
            any("non-existent" in issue for issue in c.issues) for c in claims
        )
        if claims:
            report.unsupported_claim_rate = round(report.unsupported / len(claims), 3)
        return report

    @staticmethod
    def needs_revision(report, max_unsupported_rate=0.25):
        if not report.performed or not report.total_claims:
            return False
        return (
            report.invalid_citations > 0
            or (report.unsupported_claim_rate or 0) > max_unsupported_rate
            or any(not c.numbers_ok for c in report.claims)
        )

    @staticmethod
    def claims_to_fix(report):
        return [
            c.text for c in report.claims
            if c.label == "unsupported" or not c.numbers_ok
            or any("non-existent" in i or "does not support" in i for i in c.issues)
        ]
