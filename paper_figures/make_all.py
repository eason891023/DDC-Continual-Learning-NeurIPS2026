"""Regenerate every data figure and table of the paper from results/.

Usage: python -m paper_figures.make_all
"""

from __future__ import annotations

from paper_figures import (
    fig2_mechanism,
    fig3_benchmark,
    fig4_adjudication,
    fig5_similarity,
    fig6_drift_readout,
    paper_numbers,
)


def main() -> None:
    for module in (fig2_mechanism, fig3_benchmark, fig4_adjudication, fig5_similarity, fig6_drift_readout):
        print(f"== {module.__name__}")
        module.main()
        print()
    print("== paper_figures.paper_numbers")
    paper_numbers.main()


if __name__ == "__main__":
    main()
