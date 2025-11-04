from langchain_core.prompts import PromptTemplate

CONTRADICTION_DETECTION_PROMPT = PromptTemplate(
    input_variables=["query", "passages"],
    template="""
You are an expert scientific research assistant analyzing academic papers for contradictions related to a user's query.

Your task is to identify conflicting claims in the provided passages that directly contradict each other regarding the user's question.

**Instructions:**
1. Analyze up to 10 scientific passages for contradictions related to the query: "{query}"
2. Only flag contradictions that are directly relevant to the query
3. Classify severity as "major" for direct opposition or "minor" for nuanced disagreement
4. Return a maximum of 3 most significant contradictions
5. Each contradiction must involve exactly two different passages
6. Extract concise claim texts (max 150 characters each)

**Passages:**
{passages}

**Output Format:**
Return a JSON array of contradiction objects with this exact structure:
[
  {{
    "topic": "brief topic description (max 50 chars)",
    "claim_1": {{
      "paper_id": "paper identifier",
      "text": "concise claim text (max 150 chars)"
    }},
    "claim_2": {{
      "paper_id": "different paper identifier",
      "text": "concise claim text (max 150 chars)"
    }},
    "severity": "major" or "minor"
  }}
]

If no contradictions are found, return an empty array: []

**Important:**
- Only include contradictions directly related to the query
- Ensure paper_ids are different for each contradiction
- Keep topic descriptions under 50 characters
- Truncate claim texts to 150 characters if needed
- Prioritize major contradictions over minor ones
""",
)
