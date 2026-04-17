import json
import os
from dotenv import load_dotenv

from src.agent.tools import FILE_TOOLS
from evals.executors import single_turn_executor
from evals.evaluators import tools_selected, tools_avoided, tool_selection_score
from evals.types import EvalTarget, SingleTurnResult

load_dotenv()


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_eval():
    dataset = load_dataset("evals/data/file_tools.json")
    results = []

    for i, entry in enumerate(dataset):
        data = entry["data"]
        target_data = entry["target"]

        target = EvalTarget(
            category=target_data["category"],
            expected_tools=target_data.get("expected_tools"),
            forbidden_tools=target_data.get("forbidden_tools"),
        )

        # Run the executor
        output = single_turn_executor(data, FILE_TOOLS)

        # Run evaluators based on category
        scores = {}
        if target.category == "golden":
            scores["tools_selected"] = tools_selected(output, target)
        elif target.category == "negative":
            scores["tools_avoided"] = tools_avoided(output, target)
        elif target.category == "secondary":
            scores["selection_score"] = tool_selection_score(output, target)

        results.append({
            "prompt": data["prompt"],
            "category": target.category,
            "selected": output.tool_names,
            "scores": scores,
        })

        # Print result
        status = "✓" if all(v >= 1.0 for v in scores.values()) else "✗"
        print(f"  {status} [{target.category}] {data['prompt']}")
        print(f"    Selected: {output.tool_names}")
        print(f"    Scores: {scores}")
        print()

    # Summary
    all_scores = [s for r in results for s in r["scores"].values()]
    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"Average score: {avg:.2f}")


if __name__ == "__main__":
    print("File Tools Evaluation")
    print("=" * 40)
    run_eval()
