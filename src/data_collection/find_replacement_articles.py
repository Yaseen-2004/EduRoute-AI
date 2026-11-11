
import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

API_URL = "https://en.wikipedia.org/w/api.php"

OUTPUT_FILE = Path(
    "data/processed/wikipedia/aiml_replacement_candidates.json"
)

HEADERS = {
    "User-Agent": "EduRouteAI/1.0 (B.Tech educational research project)"
}

# Topics that need better coverage in the AIML knowledge base.
REPLACEMENT_TOPICS = {
    "Explainable AI": [
        "SHAP machine learning Shapley additive explanations",
        "LIME machine learning model interpretability",
        "Explainable artificial intelligence feature attribution"
    ],
    "Generative AI": [
        "Generative adversarial network",
        "Diffusion model artificial intelligence",
        "Variational autoencoder"
    ],
    "Natural Language Processing": [
        "Natural language processing",
        "Byte pair encoding",
        "Text preprocessing machine learning"
    ],
    "MLOps": [
        "MLOps",
        "Machine learning model deployment",
        "Machine learning model monitoring"
    ],
    "Computer Vision": [
        "Image segmentation computer vision",
        "Image classification",
        "Object detection"
    ],
    "Reinforcement Learning": [
        "Q-learning",
        "Markov decision process",
        "Proximal policy optimization"
    ],
    "Machine Learning": [
        "Support vector machine",
        "Decision tree learning",
        "Random forest machine learning"
    ],
    "Deep Learning": [
        "Convolutional neural network",
        "Recurrent neural network",
        "Attention mechanism machine learning"
    ]
}


def search_wikipedia(session, query):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "format": "json",
        "utf8": 1
    }

    for attempt in range(4):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()["query"]["search"]

        except requests.RequestException as error:
            print(f"Request error: {error}")

            if attempt == 3:
                return []

            time.sleep(3 * (attempt + 1))

    return []


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    candidates = []
    seen_titles = set()

    print("\n========== EDUROUTE AI ARTICLE DISCOVERY ==========")

    for subject, queries in REPLACEMENT_TOPICS.items():

        print(f"\nSUBJECT: {subject}")
        print("=" * 65)

        for query in queries:

            print(f"\nSEARCH QUERY: {query}")

            results = search_wikipedia(session, query)

            for result in results:

                title = result.get("title", "").strip()
                normalized_title = title.casefold()

                if not title or normalized_title in seen_titles:
                    continue

                seen_titles.add(normalized_title)

                snippet = BeautifulSoup(
                    result.get("snippet", ""),
                    "html.parser"
                ).get_text(" ", strip=True)

                candidate = {
                    "title": title,
                    "subject": subject,
                    "search_query": query,
                    "snippet": snippet,
                    "source": (
                        "https://en.wikipedia.org/wiki/"
                        + requests.utils.quote(title.replace(" ", "_"))
                    ),
                    "status": "candidate_not_yet_verified"
                }

                candidates.append(candidate)

                print(f"  + {title}")

            time.sleep(2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            candidates,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n========== DISCOVERY SUMMARY ==========")
    print("Total candidates:", len(candidates))
    print("Unique titles:", len(seen_titles))
    print("Output file:", OUTPUT_FILE.resolve())
    print("Discovery completed successfully.")


if __name__ == "__main__":
    main()
