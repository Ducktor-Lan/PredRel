# Week 7 总结 — Readout Anatomy 总结与路线冻结（计划 §7）

> **状态：已完成。** 本周无新 Teacher 拟合、无模型训练、无新阈值：只把 Weeks 1–6
> 五份冻结全量证据按各自验证器逻辑逐种子重算一遍，填满计划 §7 的七个解剖字段，
> 并冻结唯一主路线。远端重算与本地重算字节一致，本地独立验证器通过。

> **最终结论（verifier 重算，非解读）：Go Route B —— Static Directed（Query/Key
> 双空间）+ simple context adapter 先行；beta-KL 优先、raw-score ranking 其次；
> DeepSets 完整上下文条件化模型暂不启用。**

本文档是 Week 7 的导航页：结论、证据位置、冻结输入、复核指南。
`evidence/` 内的远端证据不要修改；`inputs/weeks1to6/` 是随快照上服务器的
冻结输入副本（与 Weeks 1–6 本地文件逐字节一致）。

---

## 1. 七个解剖字段（原文呈现）

```text
Beyond Label: Strong — 0/6 datasets approximately uniform (Week 3 frozen gate continue_to_week04)
Beyond SupCon: Weak-at-margin — frozen gate 1/6 beyond-SupCon at 0.02 (stop_readout2rep_route); investigator override continued on 18/18 positive seed signs (p ~ 7.6e-06), Week 5 extension recheck adds 1/3
Directionality: Mixed — strong 4/9, weak 1/9, neither majority; dual-space not authorized, symmetric not sufficient alone (Week 5 frozen gate continue_to_week06_context)
Context Dependence: Moderate — 9/9 moderate, weak 0/9, strong 0/9 (Week 6 frozen gate moderate_context_adapter_candidate)
Recommended Model: Route B (Static Directed, Query/Key dual-space) with a simple context adapter first — adapter before any full context-conditioned model
Recommended Teacher Target: beta-KL first (class-residual relation), raw-score ranking second; alpha-KL retains softmax-competition confound (plan Sections 7-8)
Go / Pivot / Stop: Go Route B with adapter — not a pivot to Retrieval Anatomy, not a stop; Week 6 moderate majority is the authorizing gate
```

含义（一句话）：readout 的实例级信息是真实的（Week 3 强、Week 4 删除 6/6 忠实），
但超出 SupCon 的独有增量在冻结边际下只有弱证据（Week 4 冻结 STOP 经 investigator
override 才继续）；方向上混合（Week 5 无过半、不授权 dual-space 独占）；
上下文上一致 moderate（Week 6 9/9）。科学解读归你；本目录只保证数字可复核。

---

## 2. 最终证据（远端重算，本地已验证）

目录：`evidence/week07-anatomy-9f9544511d03-5d54a8a3/`

| 文件 | 内容 |
|---|---|
| `anatomy.json` | 全部重算记录：五个冻结门逐种子重算值、七个解剖字段、唯一路线、Week 8 交接 |
| `READOUT_ANATOMY_REPORT.md` | 人类可读报告（七字段 + 四表 + 复现记录，由脚本生成） |
| `run_input.json` / `runner.json` | 运行输入、源码 SHA、Teacher SHA、快照/运行路径 |

run_id：`week07-anatomy-9f9544511d03-5d54a8a3`；
Week 7 源码 SHA：`9f9544511d03c646d0c68b76ec7adda8158eb81a031179fa2cd85358468c7e15`；
冻结 Teacher SHA：`b6575e5b280e134b121d69c5b09d19d1dee3c72e0a231a7a8e6e6048ee643dc4`；
Week 6 授权门：`moderate_context_adapter_candidate`。

---

## 3. 冻结输入（全部逐字节校验）

`provenance/anatomy_manifest.json`（`week07-readout-anatomy-v1`）冻结了六份输入的
SHA-256；`inputs/weeks1to6/` 是随快照上服务器的逐字节副本；重算时本地 Weeks 1–6
 checkout 与快照副本任一命中即用（两者哈希一致）。

| 周 | 文件 | SHA-256 |
|---|---|---|
| 1 | Week 1/reports/week01_teacher_validation.md | `cbe2ec3b…` 全值见 manifest |
| 2 | week02-full-28262efc357e/metrics.json | `7037c54b…` |
| 3 | week03-full-b141be750774-20260912-085815/metrics.json | `fbecda4b…` |
| 4 | week04-full-3612a8d9e3b8-19198ac8/metrics.json | `9950d0db…` |
| 5 | week05-full-d07c316a09a0-f1/metrics.json | `68c8fc0d…` |
| 6 | week06-full-537520d59eaa-b05bb854/metrics.json | `1e2b4292…` |

各周源码/数据集 manifest 绑定（run_id、source SHA、manifest SHA）见
`inputs/frozen_weeks1to6.json`；每表数字均从该文件逐字拷贝。

---

## 4. 本目录结构

```text
Week 7/
├── README.md                        ← 本文件：结论、证据位置、复核指南
├── configs/server_windows.example.yaml      ← 服务器目标（<SSH_USER>@<SERVER_HOST>:<SSH_PORT>, conda base）
├── provenance/                      ← anatomy_manifest.json + teacher/week06 依赖 + source_manifest.json
├── inputs/                          ← frozen_weeks1to6.json（七字段数值源）+ weeks1to6/（冻结输入副本）
├── src/predrel_week7/               ← gates.py（四周门镜像）+ anatomy.py（重算）+ teacher_check.py
├── scripts/                         ← manifest/preflight/remote_run/render/verify + 4 个 ps1 包装
├── tests/                           ← 门镜像单测 + 端到端本地重算测试（10/10 通过）
├── evidence/                        ← ★ 远端重算证据（核心交付物）
└── reports/                         ← READOUT_ANATOMY_REPORT.template.md（仅模板）
```

服务器端（`<SSH_USER>@<SERVER_HOST>:<SSH_PORT>`，`<REMOTE_ROOT>\Week 7`）保留不可变快照
（含两个调试快照 `1e7b8388…`、`c246995d…` 与正式快照 `9f954451…`）与运行目录
（含两次失败 run：`week07-anatomy-1e7b8388a4b5-2a5f8d4f` 缺输入副本、
`week07-anatomy-c246995d5290-b14a9a4e` 缺报告渲染；均原样保留）。

---

## 5. 复核指南（怎么确认结论是真的）

1. **重跑验证器**（无需服务器，纯本地）：在 `Week 7` 下运行
   `python scripts/verify_week07_evidence.py --evidence-dir evidence/week07-anatomy-9f9544511d03-5d54a8a3 --repo-root .. --require-runner --expected-source-sha 9f9544511d03c646d0c68b76ec7adda8158eb81a031179fa2cd85358468c7e15`，
   应输出 `{"status": "pass", "mode": "anatomy", ...}`。
   验证器会重哈希六份冻结输入、按各周自有验证器逻辑逐种子重算四门、
   比对七字段与唯一路线。
2. **看报告**：`evidence/week07-anatomy-9f9544511d03-5d54a8a3/READOUT_ANATOMY_REPORT.md`
   即七字段 + Week 3/4/5/6 四表的人类可读版。
3. **看冻结值**：`inputs/frozen_weeks1to6.json` 是报告每表数字的唯一来源；
   `provenance/anatomy_manifest.json` 是全部输入哈希与路线映射的冻结记录。
4. **远端一致性**：`evidence/.../anatomy.json` 与本地快照模式重算字节一致
  （SHA-256 `38b970e2…`），run 记录绑定源码 SHA `9f954451…` 与 Teacher SHA `b6575e5b…`。
