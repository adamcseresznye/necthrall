#!/usr/bin/env python3
"""
Debug script for chunking issues.
"""

from scripts.performance_validation import PerformanceValidator
from rag.chunking import AdvancedDocumentChunker


def debug_chunking():
    validator = PerformanceValidator()

    # Test just one paper to see what's happening
    paper = validator.filtered_papers[0]
    pdf = validator.pdf_contents[0]

    print(f"Paper ID: {paper.paper_id}")
    print(f"Paper title: {paper.title}")
    print(f'Has paper_id: {hasattr(paper, "paper_id")}')
    print(f'Has title: {hasattr(paper, "title")}')
    print(f"PDF content length: {len(pdf.raw_text) if pdf.raw_text else 0}")
    print(f'PDF content preview: {pdf.raw_text[:200] if pdf.raw_text else "None"}...')
    print(f'PDF has raw_text: {hasattr(pdf, "raw_text")}')

    # Now check what the chunker validation does
    chunker = AdvancedDocumentChunker()

    try:
        chunker._validate_input(paper, pdf)
        print("Input validation passed")
    except Exception as e:
        print(f"Input validation failed: {e}")
        import traceback

        traceback.print_exc()

    # Test section detection
    try:
        print("\nTesting section detection...")
        sections = chunker._detect_sections_with_metadata(pdf.raw_text)
        print(f"Sections detected: {len(sections)}")
        if sections:
            for i, section in enumerate(sections[:3]):  # Show first 3
                print(
                    f"  Section {i}: '{section['section']}' ({len(section['content'])} chars)"
                )
        else:
            print("  No sections detected!")

        print(f"Min section chars: {chunker.min_section_chars}")
        print(f"Min chunk tokens: {chunker.min_chunk_tokens}")

    except Exception as e:
        print(f"Section detection failed: {e}")
        import traceback

        traceback.print_exc()

    # Test section-aware vs fallback chunking decision
    try:
        print("\nTesting chunking decision logic...")
        use_fallback = len(sections) < 2 or all(  # Not enough sections detected
            len(section["content"]) < chunker.min_section_chars for section in sections
        )  # All sections too short
        print(f"Use fallback chunking: {use_fallback}")
        print(f"Sections count: {len(sections)} >= 2: {len(sections) >= 2}")
        print(
            "Section lengths check:",
            all(
                len(section["content"]) >= chunker.min_section_chars
                for section in sections
            ),
        )

        if not use_fallback:
            print("Would use section-aware chunking...")

            for section in sections[:1]:  # Just test first section
                print(
                    f"Trying to chunk section '{section['section']}' ({len(section['content'])} chars)..."
                )

                # Debug sentence splitting
                print(f"  Content preview: {section['content'][:100]}...")
                sentences = chunker._split_sentences(section["content"])
                print(f"  Sentences detected: {len(sentences)}")
                if len(sentences) < 3:
                    for i, sent in enumerate(sentences):
                        print(f"    Sentence {i}: '{sent[:50]}...' ({len(sent)} chars)")

                # Test token counting
                tokens = chunker._tokenize_text(section["content"])
                print(f"  Total tokens in section: {len(tokens)}")
                print(
                    f"  Chunk size: {chunker.chunk_size}, Min tokens per chunk: {chunker.min_chunk_tokens}"
                )

                try:
                    section_chunks, section_tokens = chunker._chunk_section_with_spacy(
                        section["content"], section["section"]
                    )
                    print(
                        f"  Section chunks: {len(section_chunks)}, tokens: {section_tokens}"
                    )
                    if section_chunks:
                        print(
                            f"  First chunk content: {section_chunks[0]['content'][:50]}..."
                        )
                except Exception as e:
                    print(f"  Section chunking failed: {e}")
                    import traceback

                    traceback.print_exc()

    except Exception as e:
        print(f"Chunking decision test failed: {e}")

    # Test the actual chunking process
    try:
        print("\nTesting single paper processing (full)...")
        chunks, errors, tokens = chunker._process_single_paper_wrapper(
            paper, {pdf.paper_id: pdf}, 1
        )
        print(f"Chunks created: {len(chunks)}")
        print(f"Errors: {len(errors)}")
        print(f"Tokens: {tokens}")

        if errors:
            for error in errors:
                print(f"Error: {error}")

        if chunks:
            print(f"First chunk content: {chunks[0].content[:100]}...")

    except Exception as e:
        print(f"Chunking failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_chunking()
