import json
from dotenv import load_dotenv

from src.agent.tools import SHELL_TOOLS
from evals.executors import single_turn_executor
from evals.evaluators import tools_selected, tools_avoided, tool_selection_score
from evals.types import EvalTarget

load_dotenv()


def run_eval():
    with open("evals/data/shell_tools.json", "r") as f:
        dataset = json.load(f)

    for entry in dataset:
        data = entry["data"]
        target_data = entry["target"]

        target = EvalTarget(
            category=target_data["category"],
            expected_tools=target_data.get("expected_tools"),
            forbidden_tools=target_data.get("forbidden_tools"),
        )

        output = single_turn_executor(data, SHELL_TOOLS)

        scores = {}
        if target.category == "golden":
            scores["tools_selected"] = tools_selected(output, target)
        elif target.category == "negative":
            scores["tools_avoided"] = tools_avoided(output, target)

        status = "✓" if all(v >= 1.0 for v in scores.values()) else "✗"
        print(f"  {status} [{target.category}] {data['prompt']}")
        print(f"    Selected: {output.tool_names}  Scores: {scores}")
        print()


if __name__ == "__main__":
    print("Shell Tools Evaluation")
    print("=" * 40)
    run_eval()
