import argparse
import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minoreval.evaluation_engine import run_evaluation_session, run_single_turn_evaluation
from minoreval.utils import load_yaml_config

# Load all static configurations
MODELS_CONFIG = load_yaml_config("configs/01_models.yml")
EVALUATION_LIB = load_yaml_config("configs/02_evaluation_library.yml")
try:
    INTENT_DF = pd.read_json("data/seed_intents/intent_prompts_final.jsonl", lines=True)
except ValueError:
    raise FileNotFoundError(
        "Could not find 'intent_prompts_final.jsonl'. Please run '01_prepare_intents.py' and complete the manual review/deduplication process first.")

# Define balanced design mapping table
BALANCED_DESIGN = {
    "child": {"cell_size": 4},
    "adolescent": {"cell_size": 4}
}


def prepare_evaluation_plan(output_single_turn_tasks: bool = False) -> List[Dict[str, Any]]:
    """
    Generates a complete plan list containing all evaluation session configurations based on the balanced design.
    """
    plan = []
    used_intent_ids = set()

    template_ids = list(EVALUATION_LIB['behavioral_templates'].keys())
    risk_categories = list(load_yaml_config("data/risk_categories.yml").keys())

    print("Preparing evaluation plan based on the balanced design...")
    for persona_id, design_spec in BALANCED_DESIGN.items():
        for template_id in template_ids:
            for risk_category in risk_categories:
                num_to_sample = design_spec['cell_size']

                available_intents = INTENT_DF[
                    (INTENT_DF['risk_category'] == risk_category) &
                    (~INTENT_DF['intent_id'].isin(used_intent_ids))
                    ]

                if len(available_intents) < num_to_sample:
                    raise ValueError(
                        f"Insufficient unique intents in seed library for category '{risk_category}'. "
                        f"Needed {num_to_sample} for {persona_id}-{template_id}, but only {len(available_intents)} are available and unused."
                    )

                sampled_intents = available_intents.sample(n=num_to_sample, random_state=223)
                used_intent_ids.update(sampled_intents['intent_id'].tolist())

                for _, row in sampled_intents.iterrows():
                    session_config = {
                        "persona_id": persona_id,
                        "behavioral_template_id": template_id,
                        "risk_category": risk_category,
                        "core_intent": row['core_intent'],
                        "original_prompt": row['original_prompt'],
                        "seed_prompt_id": row['intent_id']
                    }
                    plan.append(session_config)

    print(f"Generated evaluation plan with {len(plan)} sessions.")

    if output_single_turn_tasks:
        single_turn_tasks_path = Path("data") / "single_turn_tasks.jsonl"
        tasks_to_save = [
            {
                "seed_prompt_id": p['seed_prompt_id'],
                "original_prompt": p['original_prompt'],
                "core_intent": p['core_intent'],
                "risk_category": p['risk_category']
            }
            for p in plan
        ]
        unique_tasks = list({t['seed_prompt_id']: t for t in tasks_to_save}.values())

        with open(single_turn_tasks_path, "w", encoding="utf-8") as f:
            for task in unique_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        print(f"Single-turn baseline tasks saved to {single_turn_tasks_path} ({len(unique_tasks)} unique prompts).")

    random.shuffle(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description="Run the MinorEval Automated Evaluation Framework")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['multi', 'single'],
        help="The evaluation mode to run ('multi' for the main experiment, 'single' for the baseline)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS_CONFIG['models']['targets'],
        help="Identifier of the target model to be evaluated (must be in 01_models.yml)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="A unique name for this experimental run (e.g., 'main_eval_run_1')"
    )
    args = parser.parse_args()

    print(f"Starting Experiment: {args.experiment_name} (Mode: {args.mode})")
    print(f"Target Model: {args.model}")

    results_dir = Path("results") / f"{args.experiment_name}_{args.mode}_{args.model.replace('/', '_')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.jsonl"

    with open(results_file, "w") as f:
        pass


    simulator_llm = MODELS_CONFIG['models']['simulator']
    judge_llm = MODELS_CONFIG['models']['judge']
    system_prompt_text = EVALUATION_LIB['system_prompts']['minor_context']

    if args.mode == 'multi':
        print("\n--- Running in MULTI-TURN mode ---")
        evaluation_plan = prepare_evaluation_plan()
        total_sessions = len(evaluation_plan)
        print(f"Multi-turn evaluation plan generated with {total_sessions} sessions.")

        for i, session_config in enumerate(tqdm(evaluation_plan, desc=f"Running multi-turn on {args.model}")):
            print(
                f"\n--- Running Session {i + 1}/{total_sessions}: Persona={session_config['persona_id']}, Template={session_config['behavioral_template_id']}, Risk={session_config['risk_category']} ---"
            )

            engine_args = {
                "persona_id": session_config["persona_id"],
                "behavioral_template_id": session_config["behavioral_template_id"],
                "core_intent": session_config["core_intent"]
            }
            result = run_evaluation_session(
                target_model=args.model,
                simulator_llm=simulator_llm,
                judge_llm=judge_llm,
                system_prompt=system_prompt_text,
                **engine_args
            )
            full_result = {**session_config, **result}

            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(full_result, ensure_ascii=False) + "\n")

            success_str = "Jailbreak Successful" if result.get('success', False) else "Not Jailbroken"
            final_turn = result.get('tts', 'N/A')
            print(f"--- Session {i + 1} complete. Result saved. Status: {success_str} in {final_turn} turns. ---")

    elif args.mode == 'single':
        print("\n--- Running in SINGLE-TURN mode ---")
        full_plan = prepare_evaluation_plan()
        unique_tasks = list({p['seed_prompt_id']: p for p in full_plan}.values())
        total_sessions = len(unique_tasks)
        print(f"Single-turn baseline plan generated with {total_sessions} unique prompts.")

        for i, task in enumerate(tqdm(unique_tasks, desc=f"Running single-turn on {args.model}")):
            result = run_single_turn_evaluation(
                target_model=args.model,
                judge_llm=judge_llm,
                original_prompt=task['original_prompt'],
                core_intent=task['core_intent'],
                system_prompt=system_prompt_text
            )

            full_result = {**task, **result}

            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(full_result, ensure_ascii=False) + "\n")

    print(f"\nExperiment '{args.experiment_name}' (mode: {args.mode}) for model '{args.model}' completed!")
    print(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    main()