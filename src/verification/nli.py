"""
Optional natural-language-inference check for verification (report §14.2).

The similarity, lexical and number checks cannot tell "a four-pole motor runs
at 1,500 RPM" (source) from "a six-pole motor runs at 1500 RPM" (answer): the
words and the number match. An NLI cross-encoder reads both sentences and
says whether the source entails, contradicts or is neutral towards the claim.

Enable with EDUROUTE_NLI_MODEL, e.g. cross-encoder/nli-deberta-v3-xsmall
(~70 MB; fast on a GPU, about 20-50 ms per pair on a laptop CPU).
"""

import numpy as np

LABELS = ("entailment", "neutral", "contradiction")


class NLIChecker:
    """Callable used as EvidenceMapper's `entailment_fn`.

    `judge(pairs)` takes (premise, hypothesis) pairs and returns one label per
    pair. A contradiction is only reported when the model is confident, since a
    false "contradicted" verdict would hide a correct sentence from the student.
    """

    def __init__(self, model_name, device=None, contradiction_threshold=0.95):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)
        id2label = self.model.model.config.id2label
        self.labels = [id2label[i].lower() for i in range(len(id2label))]
        if set(self.labels) != set(LABELS):
            raise ValueError(f"{model_name} is not a 3-way NLI model (labels: {self.labels})")
        self.contradiction_threshold = contradiction_threshold

    def judge(self, pairs):
        if not pairs:
            return []
        probs = np.atleast_2d(self.model.predict(pairs, apply_softmax=True, show_progress_bar=False))
        verdicts = []
        contradiction = self.labels.index("contradiction")
        for row in probs:
            label = self.labels[int(np.argmax(row))]
            if label == "contradiction" and row[contradiction] < self.contradiction_threshold:
                label = "neutral"
            verdicts.append(label)
        return verdicts

    def __call__(self, premise, hypothesis):
        return self.judge([(premise, hypothesis)])[0]


def build_nli_checker(model_name):
    """None when disabled or unavailable; verification then uses the heuristics only."""
    if not model_name:
        return None
    try:
        return NLIChecker(model_name)
    except Exception as error:  # missing download, wrong model type, ...
        import logging
        logging.getLogger(__name__).warning("NLI check disabled: %s", error)
        return None
