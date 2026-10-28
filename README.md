# MinorEval: A Behavior-driven Framework for Multi-Turn Safety Evaluation of LLMs for Minors

MinorEval is a psychologically-grounded, automated red teaming framework designed to systematically assess the dynamic safety of Large Language Models (LLMs) when interacting with minor users.

**Paper:** *MinorEval: A Behavior-driven Framework for Multi-Turn Safety Evaluation of LLMs for Minors* (Accepted to IJCNN 2026)

## 📌 Overview

Current LLM safety benchmarks predominantly rely on single-turn content filtering, which overlooks the dynamic risks stemming from interaction patterns specific to minors. MinorEval bridges this gap by replacing static scripts with **Cognitive Directives**, simulating realistic multi-turn interactions based on developmental psychology.

## 🌟 Key Features

- **Theory-Anchored Personas**: Implements age-specific psychological profiles: Child (ages 7-12, concrete operational stage) and Adolescent (ages 13-17, identity-seeking stage).
- **Behavioral Templates**: Four multi-turn attack strategies operationalized from cognitive psychology:
  - *Immersive Role-Playing* (Context Override)
  - *Goal Decomposition* (Scaffolding)
  - *Emotional Manipulation* (Helpfulness Exploit)
  - *Revisionist Questioning* (Path Drift)
- **Dynamic Simulation Engine**: Features a dual-phase prompting mechanism (Ignition & Maintenance) with a built-in **Urgency Injection** constraint to test defense resilience.
- **Rigorous Adjudication**: Automated jailbreak detection using an LLM judge with structured Rubrics and Chain-of-Thought (CoT) reasoning.

## ⚙️ Installation

```bash
git clone https://github.com/stdworld/MinorEval.git
cd MinorEval
pip install -r requirements.txt
```

## 🛠️ Configuration

### 1. API Setup
Edit `configs/01_models.yml` to configure your API key (e.g., OpenRouter) and model selections:

```yaml
openrouter:
  api_key: "YOUR_API_KEY_HERE" # Replace with your actual key
  base_url: "https://openrouter.ai/api/v1"

models:
  simulator: "qwen/qwen3-235b-a22b-2507"  # Recommended Simulator
  judge: "openai/gpt-4o-mini"              # Automated Judge
  
  targets:
    - "mistralai/mistral-7b-instruct-v0.3"
    - "openai/gpt-5-mini"
    # Add more target models to evaluate
```

### 2. Evaluation Library
Personas, behavioral templates, and system prompts are fully customizable in `configs/02_evaluation_library.yml`. Risk categories are defined in `data/risk_categories.yml`.

## 🚀 Usage

### Run Evaluations

To execute the multi-turn dynamic interaction evaluation (Main Experiment):
```bash
python scripts/02_run_evaluation.py \
  --mode multi \
  --model openai/gpt-5-mini \
  --experiment_name main_exp
```

To execute the single-turn static evaluation (Baseline):
```bash
python scripts/02_run_evaluation.py \
  --mode single \
  --model openai/gpt-5-mini \
  --experiment_name baseline_exp
```
*Outputs will be saved in `results/{experiment_name}_{mode}_{model}/results.jsonl`.*

## Project Structure

```text
MinorEval/
  configs/
    01_models.yml
    02_evaluation_library.yml
  data/
    risk_categories.yml
  minoreval/
    __init__.py
    api_handler.py
    evaluation_engine.py
    utils.py
  scripts/
    01_prepare_intents.py
    02_run_evaluation.py
  requirements.txt
  README.md
```

## 📊 Experimental Design
The framework enforces a mathematically locked paired design for fair cross-model comparison:
- **2 Personas** $\times$ **4 Attack Strategies** $\times$ **5 Risk Categories** $\times$ **4 Repetitions** = **160 Unique Scenarios**.

## 📝 Citation
If you find this framework useful for your research, please cite our paper:
```bibtex
@inproceedings{lei2026minoreval,
  title={MinorEval: A Behavior-driven Framework for Multi-Turn Safety Evaluation of LLMs for Minors},
  author={},
  booktitle={},
  year={2026},
  note={To appear}
}
```


## ⚠️ Ethical Statement
This project provides an automated red teaming framework strictly for academic research and the improvement of AI safety. The harmful scenarios simulated do not involve real human subjects or minors. Please use this repository responsibly.
