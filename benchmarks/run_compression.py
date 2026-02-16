"""
RAPTOR Compression Ratio Benchmark

Measures token compression ratio of RAPTOR's hierarchical tree structure
compared to flat chunking approaches.

The paper reports ~72% compression ratio (summaries are ~28% of child text length).

Usage:
    pip install raptor-rag
    python benchmarks/run_compression.py --input demo/sample.txt
"""

import argparse
import logging
from pathlib import Path

import tiktoken

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_compression(tree, tokenizer=None):
    """Measure compression metrics for a RAPTOR tree."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count tokens at each layer
    layer_stats = {}
    for layer_num, nodes in tree.layer_to_nodes.items():
        texts = [node.text for node in nodes]
        token_counts = [len(tokenizer.encode(text)) for text in texts]
        layer_stats[layer_num] = {
            "num_nodes": len(nodes),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_node": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
        }

    # Leaf tokens (layer 0)
    leaf_tokens = layer_stats.get(0, {}).get("total_tokens", 0)

    # Summary tokens (all non-leaf layers)
    summary_tokens = sum(stats["total_tokens"] for layer, stats in layer_stats.items() if layer > 0)

    # Total tokens across all layers
    total_tokens = sum(stats["total_tokens"] for stats in layer_stats.values())

    # Compression ratio: summary tokens / leaf tokens
    compression_ratio = summary_tokens / leaf_tokens if leaf_tokens > 0 else 0

    # Non-leaf node percentage
    total_nodes = sum(stats["num_nodes"] for stats in layer_stats.values())
    non_leaf_nodes = sum(stats["num_nodes"] for layer, stats in layer_stats.items() if layer > 0)
    non_leaf_pct = non_leaf_nodes / total_nodes * 100 if total_nodes > 0 else 0

    return {
        "layer_stats": layer_stats,
        "leaf_tokens": leaf_tokens,
        "summary_tokens": summary_tokens,
        "total_tokens": total_tokens,
        "compression_ratio": compression_ratio,
        "total_nodes": total_nodes,
        "leaf_nodes": layer_stats.get(0, {}).get("num_nodes", 0),
        "non_leaf_nodes": non_leaf_nodes,
        "non_leaf_pct": non_leaf_pct,
        "num_layers": len(layer_stats),
    }


def print_report(metrics: dict):
    """Print a formatted compression report."""
    print("\n" + "=" * 60)
    print("RAPTOR Compression Report")
    print("=" * 60)

    print("\nTree Structure:")
    print(f"  Total layers: {metrics['num_layers']}")
    print(f"  Total nodes:  {metrics['total_nodes']}")
    print(f"  Leaf nodes:   {metrics['leaf_nodes']}")
    print(f"  Summary nodes: {metrics['non_leaf_nodes']} ({metrics['non_leaf_pct']:.1f}%)")

    print("\nToken Counts:")
    print(f"  Leaf tokens:    {metrics['leaf_tokens']:,}")
    print(f"  Summary tokens: {metrics['summary_tokens']:,}")
    print(f"  Total tokens:   {metrics['total_tokens']:,}")

    print("\nCompression:")
    print(f"  Summary/Leaf ratio: {metrics['compression_ratio']:.2%}")
    print("  (Paper reports ~28% — lower is more compressed)")

    print("\nPer-Layer Breakdown:")
    for layer_num in sorted(metrics["layer_stats"].keys()):
        stats = metrics["layer_stats"][layer_num]
        print(
            f"  Layer {layer_num}: {stats['num_nodes']} nodes, "
            f"{stats['total_tokens']:,} tokens "
            f"(avg {stats['avg_tokens_per_node']:.0f}/node)"
        )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Measure RAPTOR tree compression ratio")
    parser.add_argument("--input", type=str, default="demo/sample.txt", help="Input text file")
    parser.add_argument("--tree", type=str, default=None, help="Pre-built pickled tree path")
    args = parser.parse_args()

    if args.tree:
        import pickle

        logger.info("Loading pre-built tree from %s", args.tree)
        with open(args.tree, "rb") as f:
            tree = pickle.load(f)
        metrics = measure_compression(tree)
        print_report(metrics)
    else:
        logger.info("To build a tree and measure compression:")
        logger.info("  1. Build a tree: ra.add_documents(text); ra.save('tree.pkl')")
        logger.info("  2. Run: python benchmarks/run_compression.py --tree tree.pkl")
        logger.info("  Or use demo tree: python benchmarks/run_compression.py --tree demo/cinderella")

        # Try loading demo tree
        demo_path = Path("demo/cinderella")
        if demo_path.exists():
            import pickle

            logger.info("Found demo tree at %s, measuring...", demo_path)
            with open(demo_path, "rb") as f:
                tree = pickle.load(f)
            metrics = measure_compression(tree)
            print_report(metrics)


if __name__ == "__main__":
    main()
