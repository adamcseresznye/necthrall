"""Quick test to measure actual embedding speed."""

import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("Model loaded")

# Test with 100 chunks
chunks = [
    f"This is test chunk number {i} with some scientific content about research."
    for i in range(100)
]

start = time.perf_counter()
embeddings = model.encode(
    chunks,
    batch_size=64,
    show_progress_bar=False,
    convert_to_tensor=False,
    device="cpu",
)
elapsed = time.perf_counter() - start

print(f"100 chunks in {elapsed:.3f}s = {100/elapsed:.1f} chunks/sec")
print(f"Projected 10k chunks: {10000 * elapsed / 100:.1f}s")
