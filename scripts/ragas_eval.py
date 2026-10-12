"""RAGAS evaluation of the retrieval-augmented features (precedent synthesis, copilot).

RAGAS pulls in the LangChain stack, whose versions conflict with the service's pinned dependencies, so it runs in its
own environment and only reads the cases that ``python -m ssa.cli llm-eval`` exported to
``artifacts/metrics/rag_cases.jsonl``::

    uv venv .venv-ragas --python 3.12
    uv pip install --python .venv-ragas/Scripts/python.exe -r requirements-ragas.txt   (Linux: .venv-ragas/bin/python)
    .venv-ragas/Scripts/python scripts/ragas_eval.py

Every evaluator call goes through ``ssa.agents.llm.LLMClient`` pinned to the judge model (never a generator), so the
response cache, the usage/cost ledger and ``LLM_BUDGET_USD`` apply exactly as for the rest of the evaluation.
Metrics (LLM-only, no embeddings needed): faithfulness (share of the response's claims supported by the retrieved
context; 1 - faithfulness is the hallucination rate), response groundedness and context relevance. Context precision
was left out: it costs one judge call per retrieved context (about nine per synthesis), roughly tripling the cost; answer
relevancy is measured by DeepEval. Results: ``artifacts/metrics/ragas_eval.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.llms.base import BaseRagasLLM  # noqa: E402
from ragas.metrics import ContextRelevance, Faithfulness, ResponseGroundedness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from ssa.agents.llm import LLMBudgetExceeded, LLMClient, process_spend_usd  # noqa: E402
from ssa.config import get_settings  # noqa: E402
from ssa.utils import write_json  # noqa: E402


class ClaudeJudge(BaseRagasLLM):
    """RAGAS LLM backed by the project's LLM client (judge model): cache, ledger and budget cap included."""

    def __init__(self, client: LLMClient, feature: str = "ragas"):
        super().__init__()
        self.client, self.feature = client, feature

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        text = prompt.to_string()
        gens = []
        for i in range(max(1, n)):  # n > 1 asks for variants; make each prompt distinct so the cache does not repeat one answer
            msg = text if i == 0 else f"{text}\n\n(Independent answer {i + 1}.)"
            out = self.client.chat([{"role": "user", "content": msg}], feature=self.feature, max_tokens=4000)
            gens.append(Generation(text=out.get("content") or ""))
        return LLMResult(generations=[gens])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="default: artifacts/metrics/rag_cases.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases per feature")
    args = ap.parse_args()
    s = get_settings()
    path = Path(args.cases) if args.cases else s.global_metrics_dir / "rag_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    client = LLMClient(s, model=s.judge_model)
    if not client.available:
        print("LLM not configured or LLM_ENABLED=false", file=sys.stderr)
        return 1
    metrics = [Faithfulness(), ResponseGroundedness(), ContextRelevance()]
    out = {"judge_model": s.judge_model, "cases_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
           "metrics": [m.name for m in metrics], "features": {}}
    t0 = time.time()
    try:
        for feature in sorted({r["feature"] for r in rows}):
            sub = [r for r in rows if r["feature"] == feature][: args.limit]
            ds = EvaluationDataset.from_list([{k: r[k] for k in ("user_input", "retrieved_contexts", "response")} for r in sub])
            res = evaluate(ds, metrics=metrics, llm=ClaudeJudge(client, f"ragas:{feature}"),
                           run_config=RunConfig(max_workers=4, timeout=300, max_retries=2), show_progress=False, raise_exceptions=False)
            df = res.to_pandas()
            cols = [c for c in df.columns if c in {m.name for m in metrics}]
            summary = {c: (None if df[c].isna().all() else round(float(df[c].mean()), 4)) for c in cols}
            failed = {c: int(df[c].isna().sum()) for c in cols}
            faith = df["faithfulness"].dropna() if "faithfulness" in df else None
            out["features"][feature] = {
                "n": len(sub), "mean": summary, "unscored": failed,
                "hallucination_rate": None if faith is None or faith.empty else round(float(1 - faith.mean()), 4),
                "fully_supported_rate": None if faith is None or faith.empty else round(float((faith >= 0.999).mean()), 4),
                "per_case": json.loads(df[["user_input"] + cols].to_json(orient="records"))}
            print(feature, summary, flush=True)
    except LLMBudgetExceeded as exc:
        print(f"stopped: {exc}. Nothing written.", file=sys.stderr)
        return 2
    out["seconds"] = round(time.time() - t0, 1)
    out["estimated_cost_usd"] = round(process_spend_usd(), 4)
    out["usage"] = client.usage_summary()
    write_json(s.global_metrics_dir / "ragas_eval.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "features"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
