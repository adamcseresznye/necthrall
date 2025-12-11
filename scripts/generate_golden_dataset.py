import asyncio
import json
import os
import sys
from typing import List, Dict, Any
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.semantic_scholar_client import SemanticScholarClient
from tests.eval_config import LLMJudge

TOPICS = [
    "CRISPR off-target effects",
    "Transformer architecture in NLP",
    "Climate change impact on coral reefs",
    "Quantum computing error correction",
    "mRNA vaccine technology",
]


async def generate_dataset():
    logger.info("Starting golden dataset generation...")

    # Ensure output directory exists
    output_dir = os.path.join("tests", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "golden_dataset.json")

    client = SemanticScholarClient()
    judge = LLMJudge()
    dataset = []

    for topic in TOPICS:
        logger.info(f"Processing topic: {topic}")
        try:
            # Fetch abstracts
            results = await client.multi_query_search(
                queries=[topic], limit_per_query=5
            )
            abstracts = [r.get("abstract") for r in results if r.get("abstract")]

            if not abstracts:
                logger.warning(f"No abstracts found for {topic}")
                continue

            context_text = "\n\n".join(abstracts[:5])

            # Generate Q&A
            prompt = f"""
            Based on the following labeled scientific abstracts, generate a complex scientific question and its answer.
            
            CRITICAL INSTRUCTIONS:
            1. The answer must be derived ONLY from the provided text.
            2. **Every claim in the answer must have an inline citation** in the format [1], [2], etc., corresponding to the abstract numbers below.
            3. Do not create a separate references list; use inline tags only.
            
            Abstracts:
            {context_text}
            
            Output format:
            Question: [Your question here]
            Answer: [Your answer with inline citations here]
            """

            response = await judge.a_generate(prompt)

            # Parse response (robust parsing)
            response_text = response.strip()
            question = ""
            answer = ""

            # Try to find Question and Answer blocks even with markdown formatting
            import re

            # Look for "Question:" or "**Question:**" followed by text until "Answer:" or "**Answer:**"
            q_match = re.search(
                r"(?:\*\*|)?Question:(?:\*\*|)?\s*(.*?)\s*(?:\*\*|)?Answer:(?:\*\*|)?",
                response_text,
                re.DOTALL | re.IGNORECASE,
            )
            if q_match:
                question = q_match.group(1).strip()

                # Look for "Answer:" or "**Answer:**" followed by text until end of string
                a_match = re.search(
                    r"(?:\*\*|)?Answer:(?:\*\*|)?\s*(.*)",
                    response_text,
                    re.DOTALL | re.IGNORECASE,
                )
                if a_match:
                    answer = a_match.group(1).strip()

            if not question or not answer:
                # Fallback to line-by-line if regex fails (though regex covers most cases)
                lines = response_text.split("\n")
                current_section = None
                q_lines = []
                a_lines = []

                for line in lines:
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    if "Question:" in clean_line:
                        current_section = "Q"
                        content = clean_line.split("Question:", 1)[1].strip()
                        if content:
                            q_lines.append(content)
                    elif "Answer:" in clean_line:
                        current_section = "A"
                        content = clean_line.split("Answer:", 1)[1].strip()
                        if content:
                            a_lines.append(content)
                    elif current_section == "Q":
                        q_lines.append(clean_line)
                    elif current_section == "A":
                        a_lines.append(clean_line)

                question = " ".join(q_lines).strip()
                answer = " ".join(a_lines).strip()

            if question and answer:
                entry = {
                    "input": question,
                    "actual_output": answer,
                    "retrieval_context": abstracts[:5],
                }
                dataset.append(entry)
                logger.info(f"Generated entry for {topic}")
            else:
                logger.warning(f"Failed to parse Q&A for {topic}. Response: {response}")

        except Exception as e:
            logger.error(f"Error processing {topic}: {e}")

    # Save dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Dataset saved to {output_file} with {len(dataset)} entries.")


if __name__ == "__main__":
    asyncio.run(generate_dataset())
