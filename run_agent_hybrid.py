"""CLI entrypoint for the hybrid agent.
Usage:
  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
"""
import argparse
import json
from agent.graph_hybrid import HybridAgent


def main(batch_path: str, out_path: str):
    agent = HybridAgent()
    outputs = []
    with open(batch_path, 'r', encoding='utf-8') as f:
        for line in f:
            job = json.loads(line)
            qid = job.get('id')
            question = job.get('question')
            fmt = job.get('format_hint')
            res = agent.repair_and_run(qid, question, fmt)
            out = {
                'id': qid,
                'final_answer': res.get('final_answer'),
                'sql': res.get('sql', ''),
                'confidence': res.get('confidence', 0.0),
                'explanation': res.get('explanation', '')[:200],
                'citations': res.get('citations', []),
            }
            outputs.append(out)

    with open(out_path, 'w', encoding='utf-8') as fo:
        for o in outputs:
            fo.write(json.dumps(o) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.batch, args.out)
