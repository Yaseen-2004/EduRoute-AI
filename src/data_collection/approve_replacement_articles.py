
import json
from pathlib import Path

CANDIDATE_FILE = Path(
    "data/processed/wikipedia/aiml_replacement_candidates.json"
)

EXISTING_FILE = Path(
    "data/processed/wikipedia/aiml_wikipedia_clean.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/wikipedia/aiml_approved_articles.json"
)

# Manually reviewed titles approved for initial collection.
APPROVED_TITLES = {
    "Explainable artificial intelligence",

    # Generative AI
    "Generative adversarial network",
    "Wasserstein GAN",
    "Diffusion model",
    "Autoencoder",
    "Reparameterization trick",
    "Latent diffusion model",

    # Natural Language Processing
    "Natural language processing",
    "Natural language",
    "Byte-pair encoding",

    # MLOps
    "MLOps",
    "AI observability",
    "Automated machine learning",

    # Computer Vision
    "You Only Look Once",
    "Viola–Jones object detection framework",
    "ImageNet",

    # Reinforcement Learning
    "Multi-agent reinforcement learning",
    "Partially observable Markov decision process",
    "Policy gradient method",
    "Model-free (reinforcement learning)",

    # Machine Learning
    "Relevance vector machine",
    "Structured support vector machine",
    "Least-squares support vector machine",
    "Elastic net regularization",
    "Decision tree learning",
    "Information gain (decision tree)",
    "Decision tree pruning",
    "Random forest",
    "Boosting (machine learning)",

    # Deep Learning
    "Convolutional neural network",
    "Graph neural network",
    "Deep learning",
    "Convolutional layer",
    "Recurrent neural network",
    "Bidirectional recurrent neural networks",
    "Seq2seq",
}


def load_existing_titles():
    titles = set()

    with EXISTING_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                article = json.loads(line)
                titles.add(article["title"].casefold())

    return titles


def main():
    candidates = json.loads(
        CANDIDATE_FILE.read_text(encoding="utf-8")
    )

    existing_titles = load_existing_titles()

    approved = []
    already_present = []
    not_found = []

    candidate_map = {
        item["title"].casefold(): item
        for item in candidates
    }

    for title in sorted(APPROVED_TITLES):
        key = title.casefold()

        if key in existing_titles:
            already_present.append(title)
        elif key in candidate_map:
            article = candidate_map[key].copy()
            article["status"] = "approved_for_fetching"
            approved.append(article)
        else:
            not_found.append(title)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            approved,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== APPROVAL SUMMARY ==========")
    print("Candidate records:", len(candidates))
    print("Approved for fetching:", len(approved))
    print("Already in existing dataset:", len(already_present))
    print("Approved titles not found:", len(not_found))

    print("\nAPPROVED ARTICLES:")

    for index, article in enumerate(approved, 1):
        print(
            f"{index}. [{article['subject']}] "
            f"{article['title']}"
        )

    if already_present:
        print("\nALREADY PRESENT:")
        for title in already_present:
            print("-", title)

    if not_found:
        print("\nNOT FOUND:")
        for title in not_found:
            print("-", title)

    print("\nSaved to:", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
