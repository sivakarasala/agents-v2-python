import json
from dotenv import load_dotenv

from evals.executors import multi_turn_with_mocks
from evals.evaluators import tool_order_correct, tools_avoided, llm_judge
from evals.types import MultiTurnTarget, MultiTurnResult

load_dotenv()


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_eval():
    dataset = load_dataset("evals/data/agent_multiturn.json")

    for i, entry in enumerate(dataset):
        data = entry["data"]
        target_data = entry["target"]

        target = MultiTurnTarget(
            original_task=target_data["original_task"],
            mock_tool_results=target_data.get("mock_tool_results", {}),
            category=target_data["category"],
            expected_tool_order=target_data.get("expected_tool_order"),
            forbidden_tools=target_data.get("forbidden_tools"),
        )

        # Run the executor
        output = multi_turn_with_mocks(data)

        # Run evaluators
        scores = {}
        if target.expected_tool_order:
            scores["tool_order"] = tool_order_correct(output, target)
        if target.forbidden_tools:
            scores["tools_avoided"] = tools_avoided(output, target)

        scores["output_quality"] = llm_judge(output, target)

        # Print result
        prompt = data.get("prompt", "(mid-conversation)")
        status = "✓" if all(v >= 0.7 for v in scores.values()) else "✗"
        print(f"  {status} [{target.category}] {prompt}")
        print(f"    Tools called: {output.tool_call_order}")
        print(f"    Scores: {scores}")
        print()


if __name__ == "__main__":
    print("Multi-Turn Agent Evaluation")
    print("=" * 40)
    run_eval()
