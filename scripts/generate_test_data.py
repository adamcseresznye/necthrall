import json
from typing import List

DEFAULT_URLS = [
    "https://arxiv.org/pdf/2310.06775.pdf",
    "https://arxiv.org/pdf/2310.06774.pdf",
    "https://arxiv.org/pdf/2310.06773.pdf",
    "https://arxiv.org/pdf/2310.06772.pdf",
    "https://arxiv.org/pdf/2310.06771.pdf",
    "https://arxiv.org/pdf/2310.06770.pdf",
    "https://arxiv.org/pdf/2310.06769.pdf",
    "https://arxiv.org/pdf/2310.06768.pdf",
    "https://arxiv.org/pdf/2310.06767.pdf",
    "https://arxiv.org/pdf/2310.06766.pdf",
    "https://arxiv.org/pdf/2310.06765.pdf",
    "https://arxiv.org/pdf/2310.06764.pdf",
    "https://arxiv.org/pdf/2310.06763.pdf",
    "https://arxiv.org/pdf/2310.06762.pdf",
    "https://arxiv.org/pdf/2310.06761.pdf",
    "https://arxiv.org/pdf/2310.06760.pdf",
    "https://arxiv.org/pdf/2310.06759.pdf",
    "https://arxiv.org/pdf/2310.06758.pdf",
    "https://arxiv.org/pdf/2310.06757.pdf",
    "https://arxiv.org/pdf/2310.06756.pdf",
    "https://arxiv.org/pdf/2203.02155.pdf",
    "https://arxiv.org/pdf/2106.07656.pdf",
    "https://arxiv.org/pdf/2103.14030.pdf",
    "https://arxiv.org/pdf/2005.11401.pdf",
    "https://arxiv.org/pdf/1910.01108.pdf",
]


def generate_test_data(num_papers: int = 25) -> List[str]:
    """Provides a list of PDF URLs for testing."""
    return DEFAULT_URLS[:num_papers]


if __name__ == "__main__":
    pdf_urls = generate_test_data()
    with open("test_data.json", "w") as f:
        json.dump(pdf_urls, f)
    print(
        f"Successfully generated {len(pdf_urls)} PDF URLs and saved to test_data.json"
    )
