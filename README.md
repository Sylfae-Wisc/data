# VCT 项目 — Valorant Champion Tour 预测分析系统

基于 Kaggle VCT 2021-2026 数据集，对《无畏契约》全球冠军赛进行数据分析和胜负预测的 Streamlit 平台。

- **技术栈**: Python 3.10+, pandas, scikit-learn, xgboost, statsmodels, plotly, streamlit
- **覆盖范围**: VCT 2021–2026，四大赛区（Americas / EMEA / Pacific / China）
- **四个页面**: Match Predictor / Team Dashboard / Masters London / Player Analyzer

---

## 目录

- [项目结构](#项目结构)
- [技术方案](#技术方案)
- [数据说明](#数据说明)
- [开发策略](#开发策略)
- [开始使用](#开始使用)
- [项目进度](#项目进度)

---

## 项目结构

```
├── data/
│   ├── raw/              # Kaggle 原始 CSV（已就位）
│   ├── processed/        # data_pipeline 输出
│   └── features/         # 特征工程输出
├── docs/
│   ├── 分工文档.md        # 三人分工与时间线
│   └── ...               # (记录技术决策的文档)
├── notebooks/
│   ├── 01_eda.ipynb      # 探索性数据分析
│   ├── 02_feature_eng.ipynb  # 特征工程
│   └── 03_model_compare.ipynb # 模型对比
├── src/
│   ├── data_pipeline.py  # 数据加载与清洗
│   ├── features.py       # 特征构建
│   ├── models.py         # 模型训练
│   ├── evaluation.py     # 评估指标
│   └── predictor.py      # 预测推理（供给前端三个接口）
├── app/
│   ├── main.py           # Streamlit 入口
│   ├── pages/
│   │   ├── 1_match_predictor.py   # 模板页面
│   │   ├── 2_team_analysis.py
│   │   ├── 3_masters_london.py
│   │   └── 4_player_analyzer.py
│   └── components/       # 可复用 UI 组件
├── tests/
├── .gitignore
├── README.md
├── 项目进度.md
└── requirements.txt
```

---

## 技术方案

### 模型

| 模型 | 用途 |
|------|------|
| XGBoost（主力） | 胜负概率预测 |
| Elo Rating | 队伍强度基线 |
| Logistic Regression | 概率校准辅助 |

### BP 预测

基于历史 Ban/Pick 统计的条件概率：
- P(ban X | team 选了 Y) — 对方选了某英雄后 ban X 的概率
- Pick 推荐基于剩余可用英雄池

### BO3 比分

链式乘法计算：

```
P(2-0) = P(map1 win) × P(map2 win)
P(2-1) = P(map1 win) × P(map2 lose) × P(map3 win)
        + P(map1 lose) × P(map2 win) × P(map3 win)
```

### 特征

ACS, KAST, ADR, HS%, 历史对战记录, 近期 form（最近 N 场）, 地图池, 英雄阵容

### 评估指标

| 指标 | 用途 |
|------|------|
| Brier Score / Log Loss | 概率校准质量 |
| ROC-AUC | 区分能力 |
| Accuracy / F1 | 分类准确率 |
| McNemar's Test | 模型间显著性检验 |
| Walk-forward CV | 时序验证（防止数据泄漏） |

---

## 数据说明

### 数据来源

Kaggle: [ryanluong1/valorant-champion-tour-2021-2023-data](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)

数据按年份 `vct_2021 ~ vct_2026` 组织，每年结构相同。另含全局 ID 表。

```
data/raw/
├── all_ids/              # 全局 ID 映射（跨年份）
│   ├── all_matches_games_ids.csv   # 比赛-地图-游戏 ID
│   ├── all_players_ids.csv         # 选手 ID
│   ├── all_teams_ids.csv           # 战队 ID
│   ├── all_teams_mapping.csv       # 简称↔全称
│   └── all_tournaments_stages_match_types_ids.csv
└── vct_202X/
    ├── agents/
    │   ├── agents_pick_rates.csv   # 英雄选取率
    │   ├── maps_stats.csv          # 地图攻防胜率
    │   └── teams_picked_agents.csv # 战队英雄使用
    ├── ids/
    ├── matches/
    │   ├── overview.csv            # 选手单场数据（核心建模表）
    │   ├── scores.csv              # 比赛比分（目标变量）
    │   ├── maps_scores.csv         # 地图级比分
    │   ├── draft_phase.csv         # BP 记录（ban/pick）
    │   ├── kills*.csv, eco*.csv   # 击杀/经济明细
    │   └── ...                     # 其他详情表
    └── players_stats/
        └── players_stats.csv       # 选手赛事级汇总（25 字段）
```

### 核心建模表

| 表 | 路径 | 用途 |
|----|------|------|
| **overview** | `matches/overview.csv` | 选手级特征（ACS/KAST/ADR/HS%） |
| **scores** | `matches/scores.csv` | 比赛比分（目标变量） |
| **players_stats** | `players_stats/players_stats.csv` | 选手赛事级汇总 |
| **draft_phase** | `matches/draft_phase.csv` | BP 记录（ban/pick） |
| **agents_pick_rates** | `agents/agents_pick_rates.csv` | 英雄选取率 |

### 关键字段

| 缩写 | 全称 | 含义 |
|------|------|------|
| ACS | Average Combat Score | 平均战斗评分 |
| KAST | Kill/Assist/Survive/Trade % | 选手参与回合比例 |
| ADR | Average Damage Per Round | 平均每回合伤害 |
| HS% | Headshot % | 爆头率 |
| KD | Kills - Deaths Difference | 击杀/死亡差 |
| KPR | Kills Per Round | 每回合击杀数 |
| FKPR | First Kills Per Round | 每回合首杀数 |
| Rating | VCT Rating | VCT 官方综合评分 |

### 数据使用说明

- 数据路径 `data/raw/vct_202X/`，由 `data_pipeline.py` 加载到 `data/processed/`
- `all_ids/` 表跨年份全局可用
- 每张 CSV 前 5 列（Tournament / Stage / Match Type / Match Name / Map）均为标识字段，用于表间关联

### CSV Schema（所有年份通用）

<details>
<summary>全局 ID 表（all_ids/）</summary>

| 文件 | 字段 | 用途 |
|------|------|------|
| `all_matches_games_ids` | Tournament, Tournament ID, Stage, Stage ID, Match Type, Match Type ID, Match Name, Match ID, Map, Game ID, Year | 全局比赛-地图 ID 映射 |
| `all_players_ids` | Player, Player ID | 选手 ID 映射 |
| `all_teams_ids` | Team, Team ID | 战队 ID 映射 |
| `all_teams_mapping` | Abbreviated, Full Name | 战队简称↔全称 |
| `all_tournaments_stages_match_types_ids` | Tournament, Tournament ID, Stage, Stage ID, Match Type, Match Type ID, Year | 赛事-阶段-类型 ID |

</details>

<details>
<summary>agents/ — 英雄数据</summary>

| 文件 | 字段 | 说明 |
|------|------|------|
| agents_pick_rates | Tournament, Stage, Match Type, Map, Agent, Pick Rate | 英雄在各图的选取率 |
| maps_stats | Tournament, Stage, Match Type, Map, Total Maps Played, Attacker Win %, Defender Win % | 地图攻防胜率 |
| teams_picked_agents | Tournament, Stage, Match Type, Map, Team, Agent, Total Wins, Total Loss, Total Maps Played | 战队英雄使用战绩 |

</details>

<details>
<summary>matches/ — 比赛数据（核心）</summary>

| 文件 | 字段 | 说明 |
|------|------|------|
| overview | Tournament, Stage, Match Type, Match Name, Map, Player, Team, Agents, Rating, ACS, Kills, Deaths, Assists, KD, Survive %, ADR, HS%, First Kills, First Deaths, Side | 选手单场数据 |
| scores | Tournament, Stage, Match Type, Match Name, Team A, Team B, Team A Score, Team B Score, Match Result | 比赛比分（目标变量） |
| maps_scores | Tournament, Stage, Match Type, Match Name, Map, Team A, Team A Score, Team A Attacker/Defender/Overtime Score, Team B, Team B Score, Team B Attacker/Defender/Overtime Score, Duration | 地图级比分 |
| draft_phase | Tournament, Stage, Match Type, Match Name, Team, Action, Map | BP 记录 |
| maps_played | Tournament, Stage, Match Type, Match Name, Map | 比赛地图列表 |
| eco_rounds | Tournament, Stage, Match Type, Match Name, Map, Round Number, Team, Loadout Value, Remaining Credits, Type, Outcome | 回合经济 |
| eco_stats | Tournament, Stage, Match Type, Match Name, Map, Team, Type, Initiated, Won | 经济局统计 |
| kills | Tournament, Stage, Match Type, Match Name, Map, Player Team, Player, Enemy Team, Enemy, Player Kills, Enemy Kills, Difference, Kill Type | 击杀明细 |
| kills_stats | Tournament, Stage, Match Type, Match Name, Map, Team, Player, Agents, 2k/3k/4k/5k, 1v1~1v5, Econ, Spike Plants, Spike Defuses | 击杀/残局统计 |
| rounds_kills | Tournament, Stage, Match Type, Match Name, Map, Round Number, Eliminator Team, Eliminator, Eliminator Agent, Eliminated Team, Eliminated, Eliminated Agent, Kill Type | 回合击杀 |
| team_mapping | Abbreviated, Full Name | 战队简称↔全称 |
| win_loss_methods_count | Tournament, Stage, Match Type, Match Name, Map, Team, counts by Elimination/Detonated/Defused/Time Expiry | 获胜方式统计 |
| win_loss_methods_round_number | Tournament, Stage, Match Type, Match Name, Map, Round Number, Team, Method, Outcome | 回合胜负方式 |

</details>

<details>
<summary>players_stats/ — 选手综合统计</summary>

| 文件 | 字段 | 说明 |
|------|------|------|
| players_stats | Tournament, Stage, Match Type, Player, Teams, Agents, Rounds Played, Rating, ACS, K:D, Survive %, ADR, KPR, APR, FKPR, FDPR, HS%, Clutch Success %, Clutches, Max Kills, Kills, Deaths, Assists, First Kills, First Deaths | 选手赛事级汇总（25 字段） |

</details>

---

## 开发策略

先搭地基（数据→特征→模型→预测接口），再做 Match Predictor 作为**模板页面**，其余两人照着复制。

```
Phase 0           Phase 1           Phase 2            Phase 3
Person A(地基) →  Person B(模板页) → Person C+A(复制) → 三人(收尾)
```

详请见 [docs/分工文档.md](docs/分工文档.md)。

---

## 开始使用

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Streamlit 应用
streamlit run app/main.py

# 运行测试
pytest tests/
```

---

## 项目进度

当前进展和待做事项见 [项目进度.md](项目进度.md)。
