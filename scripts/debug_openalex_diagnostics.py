"""Run OpenAlex API diagnostic helpers from SearchAgent.

This script calls debug_filter_combinations, analyze_website_equivalent_query,
and check_date_filtering for a representative query and prints logs.
"""

from agents.search import SearchAgent


def main():
    agent = SearchAgent()
    q = (
        "persistent organic pollutants OR POPs OR environmental risk OR ecotoxicity "
        "OR bioaccumulation OR environmental fate OR chemical safety"
    )

    print("Running OpenAlex diagnostics for query:\n", q)

    try:
        agent.debug_filter_combinations(q)
    except Exception as e:
        print("debug_filter_combinations failed:", e)

    try:
        agent.analyze_website_equivalent_query(q)
    except Exception as e:
        print("analyze_website_equivalent_query failed:", e)

    try:
        agent.check_date_filtering(q)
    except Exception as e:
        print("check_date_filtering failed:", e)


if __name__ == "__main__":
    main()
