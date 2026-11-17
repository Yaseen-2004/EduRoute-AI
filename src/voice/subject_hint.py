"""
Which subject is a spoken question about? Used for a second Whisper pass with that
subject's vocabulary when the student has not chosen a subject.

The first-pass transcript is searched in the knowledge-base index the app already
has loaded (one query embedding, ~30 ms), and the top passages vote for a subject.
A weak match (below the router's in-domain level) gives no subject, so an
off-syllabus question is not pushed towards engineering vocabulary.
"""

from collections import Counter

MIN_SIMILARITY = 0.45      # a little below the router's calibrated in-domain threshold (0.491)


def make_subject_finder(retriever, catalogue, top_k=5):
    subjects = {s["id"]: s for s in catalogue["subjects"]}

    def find_subject(text):
        try:
            hits = retriever.retrieve(text, top_k=top_k)
        except Exception:
            return None
        if not hits or (hits[0].get("similarity_score") or 0) < MIN_SIMILARITY:
            return None
        votes = Counter()
        for rank, hit in enumerate(hits):
            if hit.get("subject_id") in subjects:
                votes[hit["subject_id"]] += 1 / (rank + 1)           # higher-ranked passages count more
        if not votes:
            return None
        subject = subjects[votes.most_common(1)[0][0]]
        return subject["name"], subject["topics"]

    return find_subject
