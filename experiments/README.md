# Experiments

本目录保存不参与机器人生产启动的离线实验和验收工具。

- `manifests/`：冻结问题、事实槽和审计样本。
- `memory_foreground_benchmark.py`：前台检索端到端基准。
- `memory_foreground_*_ab.py`：原子查询、追问规划和预压缩消融。
- `roleplay_*`：角色扮演回答与私人记忆边界验收。
- `memory_domain_smoke.py`、`virtual_multi_account_acceptance.py`：记忆域和多账号隔离验收。
- `memory_growth_smoke.py`：后台增长与并发读取验收。
- `candidate_*`、`multi_candidate_*`：Association 候选增长和跨查询收益实验。
- `bge_reranker_candidate_ab.py`：BGE reranker 候选池消融。

从项目根目录使用模块方式执行：

```bash
python -m experiments.memory_foreground_benchmark --help
python -m experiments.roleplay_acceptance_test --help
```

实验输出写入 `logs/`，不会被机器人运行时加载。
