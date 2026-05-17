from doc_qual import compute_doc_qual_score


def main() -> None:
    result = compute_doc_qual_score("path/to/document.jpg", verbose=False)
    print(f"Score: {result.ocr_score:.1f}")
    print(f"Passed: {result.passed}")
    for recommendation in result.recommendations:
        print(f"- {recommendation}")


if __name__ == "__main__":
    main()
