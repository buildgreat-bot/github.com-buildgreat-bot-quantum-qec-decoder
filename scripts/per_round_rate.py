"""Convert multi-round logical error rates to approximate per-round error rates.

Formula:
    p_round = (1 - (1 - 2 * p_total) ** (1 / rounds)) / 2

NOTE: This is an analytical approximation assuming independent, identically
distributed (i.i.d.) logical flips across rounds without temporal correlations.
"""

import sys
import argparse


def calculate_per_round_rate(p_total: float, rounds: int = 5) -> float:
    """Computes approximate per-round flip probability assuming i.i.d. rounds."""
    if p_total < 0.0 or p_total >= 0.5:
        raise ValueError(f"p_total must be in [0, 0.5), got {p_total}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    return (1.0 - (1.0 - 2.0 * p_total) ** (1.0 / rounds)) / 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Convert total multi-round logical error rate to approximate per-round error rate."
    )
    parser.add_argument(
        "--p-total",
        type=float,
        default=0.0579,
        help="Total per-observable logical error rate (default: 0.0579, matching [[18,4,4]] 5-round benchmark)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of syndrome extraction rounds (default: 5)",
    )
    args = parser.parse_args()

    p_round = calculate_per_round_rate(args.p_total, args.rounds)

    print("=================================================================")
    print("      Multi-Round to Per-Round Logical Error Rate Converter      ")
    print("=================================================================")
    print(f"  * Total Logical Error Rate (p_total) : {args.p_total:.4f} ({args.p_total:.2%})")
    print(f"  * Extraction Rounds (R)              : {args.rounds}")
    print(f"  * Approx. Per-Round Rate (p_round)   : {p_round:.6f} ({p_round:.4%})")
    print("-----------------------------------------------------------------")
    print("  [!] Model Assumption: Independent, identically distributed (i.i.d.)")
    print("      per-round logical observable flips without temporal correlations.")
    print("=================================================================")


if __name__ == "__main__":
    main()
