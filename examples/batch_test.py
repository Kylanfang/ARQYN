# encoding: utf-8
"""ARQYN v2 M6：无副作用离线批测（用户独立判断入口）。

用法（任意目录可跑）：
  <staging>/decision-venv/Scripts/python.exe \
      <lock_dir>/batch_test.py \
      --cases <lock_dir>/cases.jsonl \
      --out   <lock_dir>/cases_result.jsonl

实现：克隆真实分区记录为 schema 壳（state/questions 结构与训练评测逐字段同源），
复用产品 evaluate CLI（部署同构装载）——不执行任何真实控件、不写产品仓。
cases.jsonl 每行：{case_id, user_question, state{条件字段}, candidates?(缺省=产品冻结召回), user_expected?(用户先写自己的判断)}
"""
import argparse, ast, json, os, subprocess, sys, tempfile, datetime

sys.path.insert(0, os.environ.get("ARQYN_REPO", "."))  # product repo root

VENV_PY = r"<staging>/decision-venv/Scripts/python.exe"
TEMPLATE = r"<v2_dir>/data/generated/partitions/calib.jsonl"
CATALOG = r"<v2_dir>/data/operation_catalog_v2.json"
LOCKDIR = r"<lock_dir>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=r"<adapter_dir>")
    ap.add_argument("--hf-home", default=r"<staging>/hf-home")
    a = ap.parse_args()

    tpl = ast.literal_eval(json.loads(open(TEMPLATE, encoding="utf-8").readline())["state"] 
                           if isinstance(json.loads(open(TEMPLATE, encoding="utf-8").readline())["state"], str)
                           else repr(json.loads(open(TEMPLATE, encoding="utf-8").readline())["state"]))
    tpl_full = json.loads(open(TEMPLATE, encoding="utf-8").readline())
    tpl_q = ast.literal_eval(tpl_full["questions"]) if isinstance(tpl_full["questions"], str) else tpl_full["questions"]
    tpl_state = ast.literal_eval(tpl_full["state"]) if isinstance(tpl_full["state"], str) else tpl_full["state"]
    _cat = json.load(open(CATALOG, encoding="utf-8"))
    _ops = _cat["operations"] if isinstance(_cat, dict) else _cat
    catalog = {o["operation_id"]: o for o in _ops}
    from uav.decision.candidates import shortlist
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]

    tmp = os.path.join(LOCKDIR, "_cases_partition_tmp.jsonl")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(cases):
            _sl = shortlist(c["user_question"], c.get("current_page", "/workbench"), top_k=8)
            cands = c.get("candidates") or [x["operation"] if isinstance(x, dict) and "operation" in x
                                            else x.get("operation_id", str(x)) for x in _sl]
            criteria = {}
            for op in cands:
                m = catalog.get(op, {})
                criteria[op] = m.get("action") or m.get("effect") or op
            state = dict(tpl_state)
            state["user_request"] = c["user_question"]
            state["current_page"] = c.get("current_page", "/workbench")
            ps = dict(state.get("project_state") or {})
            for k, v in (c.get("state") or {}).items():
                ps[k] = v
            state["project_state"] = ps
            q = {"operation": {"type": "choice",
                               "instructions": tpl_q["operation"]["instructions"],
                               "criteria": criteria},
                 "clarify": tpl_q["clarify"]}
            gold_ops = c["user_expected"] if isinstance(c.get("user_expected"), list) else (
                [c["user_expected"]] if c.get("user_expected") else [])
            rec = {"schema_version": tpl_full["schema_version"],
                   "sample_id": c.get("case_id", "case-%03d" % i), "root_id": c.get("case_id", "case-%03d" % i),
                   "variant": "batch", "category": c.get("category", "batch"),
                   "state": repr(state), "questions": repr(q),
                   "gold": repr({"operations": gold_ops, "needs_clarification": False}),
                   "annotation": repr({"source": "user_batch_test", "condition_class": "unknown", "status": "batch"}),
                   "provenance": repr({"generator": "lock/batch_test.py", "note": "M6 用户独立判断入口"})}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    recs_out = os.path.join(LOCKDIR, "_cases_eval_tmp.jsonl")
    m_out = os.path.join(LOCKDIR, "_cases_metrics_tmp.json")
    cmd = [VENV_PY, "-m", "uav.training.masked_qlora", "evaluate",
           "--data", tmp, "--out", recs_out, "--adapter", a.adapter,
           "--metrics-out", m_out, "--hf-home", a.hf_home]
    r = subprocess.run(cmd, cwd=r"<repo>", capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        sys.exit(r.returncode)

    cby = {c.get("case_id", "case-%03d" % i): c for i, c in enumerate(cases)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as out:
        for line in open(recs_out, encoding="utf-8"):
            e = json.loads(line)
            sid = e.get("sample_id")
            c = cby.get(sid, {})
            out.write(json.dumps({
                "case_id": sid,
                "user_question": c.get("user_question"),
                "state": c.get("state", {}),
                "candidates_order": ast.literal_eval(json.loads(
                    json.dumps(e.get("input", {}).get("choices", "[]"))) ) if isinstance(e.get("input"), dict) else None,
                "user_expected": c.get("user_expected", None),
                "model_top1": e.get("prediction") or e.get("top1") or e.get("pred"),
                "record": e,
                "executed": False,
                "note": "离线批测：未执行任何真实控件；顺序敏感为已知局限（固定候选序合同）",
            }, ensure_ascii=False) + "\n")
    for p in (tmp, recs_out):
        try: os.remove(p)
        except OSError: pass
    sys.stderr.write(json.dumps({"ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "n_cases": len(cases),
                                 "caution": "结果含原始评测记录字段名（prediction/…），未套校准温度时置信度为原始分"},
                                ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
