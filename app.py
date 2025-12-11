from analyzer import analyze_document
from aggregator import aggregate_results

if __name__ == "__main__":
    path = "sample_docs/doc2 - Non-MNPI.docx"
    results = analyze_document(path)
    summary = aggregate_results(results)

    print("\nFinal Summary:")
    print(summary)
