# VCT 项目

基于 Kaggle VCT 2021-2026 数据集的比赛胜负预测与数据分析平台。

---

## 项目结构

```
├── data/
│   ├── raw/              # Kaggle 原始 CSV（已就位）
│   ├── processed/        # data_pipeline 输出
│   │   ├── match_dataset.parquet   # 12478 场 × 34 列（匹配级）
│   │   ├── draft_dataset.parquet   # 19164 条 BP 记录
│   │   └── team_stats.parquet      # 队伍聚合统计
│   └── features/         # 特征工程输出
│       └── match_features.parquet  # 12478 × 55（21 个特征列）
├── docs/
│   ├── 分工文档.md
│   └── 项目进度.md
├── models/
│   ├── xgb_model.pkl              # 训练好的 XGBoost
│   ├── imputer.pkl                # 均值填充器
│   ├── feature_importance.csv     # 特征重要性排序
│   └── cv_results.csv             # Walk-forward CV 结果
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   └── 03_model_compare.ipynb
├── src/
│   ├── data_pipeline.py  # 数据加载与清洗（✅）
│   ├── features.py       # 特征构建（✅）
│   ├── models.py         # 模型训练（✅）
│   ├── evaluation.py     # 评估指标（✅）
│   └── predictor.py      # 预测推理（❌ 待做）
├── app/
│   ├── main.py           # Streamlit 入口（❌ 待做）
│   ├── pages/
│   │   ├── 1_match_predictor.py   # 模板页面（❌ 待做）
│   │   ├── 2_team_analysis.py     # （❌ 待做）
│   │   ├── 3_masters_london.py    # （❌ 待做）
│   │   └── 4_player_analyzer.py   # （❌ 待做）
│   └── components/       # 可复用 UI 组件（❌ 待做）
├── tests/                # pytest（❌ 待做）
├── requirements.txt      # ✅ 已定
├── .gitignore
└── README.md
```

---

## 当前进度

| Phase | 任务 | 状态 |
|-------|------|------|
| 0-1 | 环境搭建（requirements + 安装） | ✅ |
| 0-2 | 数据下载 + Schema 记录 | ✅ |
| 0-3 | data_pipeline（加载→清洗→合并→输出 parquet） | ✅ |
| 0-4 | features / models / evaluation | ✅ |
| 0-5 | predictor.py 三个预测接口 | ❌ |
| 1 | Streamlit 模板页（main + Match Predictor） | ❌ |
| 2 | 按模板扩展三个页面 | ❌ |
| 3 | 测试 + 收尾 | ❌ |

完整进度见[docs/项目进度.md](docs/项目进度.md)。

---

## 技术方案

### 模型

| 模型 | 用途 |
|------|------|
| XGBoost（主力） | 胜负概率预测，Walk-forward CV 验证 |
| Logistic Regression | 概率校准辅助 |

### Walk-forward CV 结果

| 测试年份 | XGBoost Brier | XGBoost AUC | Logistic Brier | Logistic AUC |
|----------|--------------|-------------|----------------|--------------|
| 2023 | 0.050 | 0.984 | 0.053 | 0.981 |
| 2024 | 0.067 | 0.971 | 0.069 | 0.970 |
| 2025 | 0.069 | 0.969 | 0.068 | 0.969 |
| 2026 | 0.072 | 0.966 | 0.079 | 0.958 |

### BP 预测
基于历史 Ban/Pick 统计的条件概率：P(ban X | team 选了 Y)

### BO3 比分
链式乘法：P(2-0) = P(map1 win) × P(map2 win)

### 特征设计

| 类别 | 特征 | 说明 |
|------|------|------|
| Form | form_5/10/20（滚动窗口） | 队伍近 N 场胜率，含差值 |
| H2H | h2h_ratio | 两队历史交手记录 |
| Stat Diff | diff_kd / diff_acs / diff_adr / 等 7 项 | 团队统计量差值 |

**注意**：diff 系特征来自 overview 表（赛后统计），赛中可用但赛前不可知。predictor 将分两条路径处理。

### 评估
Brier Score / Log Loss / ROC-AUC / F1 / McNemar's Test / ECE / Walk-forward CV

### 数据

6 年 VCT 数据（2021-2026），按年份 `vct_202X/` 组织，核心表：

| 表 | 用途 |
|----|------|
| overview | 选手级特征（ACS/KAST/ADR/HS%） |
| scores | 比赛比分（目标变量） |
| players_stats | 选手赛事级汇总 |
| draft_phase | BP 记录（ban/pick） |
| agents_pick_rates | 英雄选取率 |

完整 Schema 见 [README.md](README.md)。

### 管道技术细节

- **merge 逻辑**：scores.csv（match 级）与 overview.csv（player-match-map 级）通过 Tournament/Stage/Match Type/Match Name/Year + Team 关联
- **聚合方式**：overview 先聚合到 match-team 级别（跨选手平均），再分拆 team_a / team_b
- **百分比处理**：KAST、HS% 原始格式为 `"78%"`，自动 strip `%` → `/100`
- **空值**：约 25%（rating, kast）— SimpleImputer(mean) 填充
- **数据泄漏**：diff_kd（重要性 53%）、diff_acs（16%）来自本场统计，用作赛前预测需分离

---

## 开发流程

### 运行命令

```bash
pip install -r requirements.txt
python src/data_pipeline.py          # 运行数据管道
python src/features.py               # 构建特征
python src/models.py                 # 训练模型
streamlit run app/main.py            # 启动 UI
pytest tests/                        # 运行测试
```

### 分支

- `main` — 稳定版本
- `feature/*` — 功能开发
- `fix/*` — 修复

---

## 代码约定

- **页面三段式**: 数据加载 → 计算 → 渲染，`# ====` 分隔
- **组件复用**: 先看 `app/components/` 有无现成组件
- **数据路径**: 统一 `DATA_DIR = "data/"` 常量
- **模型接口**: 每个模型实现 `train()` 和 `predict()`
- **类型注解**: src/ 下的函数需 type hints

### predictor.py 接口

```python
predict_match(team1, team2)       → {"team1_win_prob": 0.62, ...}
predict_bp(team1, team2)          → {"ban_prob": {...}, "pick_prob": {...}}
predict_bo3_score(team1, team2)   → {"2-0": 0.25, "2-1": 0.30, ...}
```

页面开发者只调这三个函数，不直接加载模型。

### 验证

- 改 pipeline 后跑 `pytest tests/`
- 改特征后对比 Brier Score 不劣化
- 改 UI 后在本地预览

---

## 红线上报（先问我）

删除文件/git历史、修改密钥/CI、git push/reset/rebase、装全局依赖、公开发布。
