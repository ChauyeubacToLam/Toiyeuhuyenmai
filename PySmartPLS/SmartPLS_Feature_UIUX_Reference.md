# SmartPLS 3 & 4 — Feature & UI/UX Reference for PySmartPLS port

> A complete, implementation-oriented specification + checklist for building a Python/PySide6 PLS-SEM application that matches and exceeds SmartPLS 3 and SmartPLS 4.
> Treat every table row as an acceptance criterion. Where a behavior is "exact" (e.g., canvas interaction, color thresholds), replicate it literally.

---

## 1. Executive Summary — SmartPLS 3 vs SmartPLS 4

### 1.1 What SmartPLS 3 offers
SmartPLS 3 is a mature, **PLS-only** path-modeling tool (composite-based SEM):
- **Estimators:** PLS-SEM Algorithm, Consistent PLS (PLSc), Bootstrapping, Consistent Bootstrapping, **Blindfolding (Q²)**, PLSpredict, CTA-PLS, IPMA, PLS-MGA, Permutation/MICOM, FIMIX-PLS, PLS-POS.
- **No covariance-based SEM, no CFA, no regression module.**
- **Editable convergence settings** (Max iterations default 300, stop criterion entered as exponent X in 10⁻ˣ).
- **Centroid weighting scheme** available; PCA not available.
- **Light theme only** (Java/Swing look), MDI windows (one model per window, switched via a `Window` menu).
- **CSV-centric import**; model+data stored in project folders/zips (legacy `.splsm`).
- Reports are **not savable/reopenable or comparable** side-by-side.
- Bootstrap column label: **`Standard Error (STERR)`**, `T Statistics (|O/STERR|)`. **Random seed only** (no reproducibility).

### 1.2 What SmartPLS 4 adds / changes
SmartPLS 4 is described in release notes as a **"fundamentally renewed and improved" GUI** and a full multi-method suite:
- **New estimators:** CB-SEM + CFA (ML), GSCA, Regression, Logistic Regression, **Path Analysis & PROCESS** (Hayes-style, beta), **NCA + NCA Permutation**, **CVPAT**, endogeneity via **Gaussian Copulas**, **Prediction-Oriented Model Selection / Model Comparison (BIC)**, CB-SEM Bootstrapping/MGA/MICOM/Moderation, Consistent MGA/Permutation.
- **Blindfolding REMOVED** (replaced by PLSpredict + CVPAT for predictive assessment).
- **Convergence settings FIXED** (Max iterations = 3,000, stop criterion = 1.0E-7, non-editable).
- **Weighting:** Centroid removed; **PCA added** (Path default, Factor, PCA).
- **Workspace folder** concept (local or cloud), 5 context views including a **Report Comparison / Compare View**.
- **Native Excel (.xlsx) and SPSS (.sav) import**; **create data files from calculation results** (eases two-stage HOC).
- **Saveable/reopenable/exportable reports** (Excel/HTML/R), side-by-side comparison.
- **Per-element styling** (shapes circle/rectangle/hexagon/octagon, colors, borders, fonts), connection styling (dotted/dashed/solid, decorations, corner routing), **Dark mode + color-blind themes**, grid snap + green-ruler positioning helper, Markdown/image comment nodes.
- Bootstrap relabel: **`Standard Deviation (STDEV)`**, `T statistics (|O/STDEV|)`; **Fixed seed** added; BCa/studentized CI fixed; parallel processing; "Most important (faster)" vs "Complete (slower)".
- **Target version for the port = SmartPLS 4** (use rho_a / rho_c naming, fixed convergence, Compare View, theming).

---

## 2. MASTER FEATURE CHECKLIST

> Version key: **3** = SmartPLS 3, **4** = SmartPLS 4, **3+4** = both. Priority: **P0** core (MVP), **P1** important, **P2** advanced.

### 2.1 Project / data / workspace
| Feature | Ver | UI Location | Priority |
|---|---|---|---|
| Workspace folder (local/cloud) | 4 | First-launch picker | P1 |
| New project (name dialog) | 3+4 | Files menu / toolbar; Project Explorer | P0 |
| Import CSV (delimiter, missing marker) | 3+4 | Import dialog / Data View | P0 |
| Import Excel `.xlsx` / SPSS `.sav` directly | 4 | Import dialog | P1 |
| Import project from backup `.zip` | 3+4 | Files > Import project from backup | P1 |
| Data View spreadsheet + descriptives | 3+4 | Data View | P0 |
| Define Data Groups (Add / Generate) | 3+4 | Data View toolbar | P1 |
| Create data file from calculation results | 4 | Results > Create data file | P1 |
| Sample/example projects (categorized) | 4 | Project View | P2 |

### 2.2 Modeling / canvas
| Feature | Ver | UI Location | Priority |
|---|---|---|---|
| Modeling View (indicators left, canvas right) | 3+4 | Modeling View | P0 |
| Drag-drop indicators → new construct | 3+4 | Canvas | P0 |
| Drag-drop indicators → existing construct | 3+4 | Canvas | P0 |
| Drawing mode (click empty → lv0/lv1…) | 3+4 | Toolbar + canvas | P0 |
| Connect tool (construct→construct path) | 3+4 | Toolbar + canvas | P0 |
| Invert measurement model / Reverse Link (Ctrl+R) | 3+4 | Right-click construct | P0 |
| Double-click construct → mode A/B dialog | 3+4 | Construct | P0 |
| Delete element (data preserved) | 3+4 | Delete key / Ctrl+D | P0 |
| Move construct moves its indicators | 3+4 | Canvas | P0 |
| Hide / Show indicators (Ctrl+H / Ctrl+V) | 3+4 | Right-click construct | P1 |
| Arrange indicators above/below/left/right | 3+4 | Right-click; Ctrl+I/J/K/L | P1 |
| Status coloring (red invalid / blue valid) + tooltip | 3+4 | Canvas | P0 |
| Used-indicator red font in list | 3+4 | Indicators list | P0 |
| Group/Ungroup, front/back | 3+4 | Toolbar; Ctrl+G/U/F/B | P1 |
| Zoom (wheel) / pan (Ctrl+drag) | 3+4 (pan=4) | Toolbar/mouse | P1 |
| Grid + snap-to-grid | 3+4 (snap=4) | Toolbar; Ctrl+T | P2 |
| Green-ruler positioning helper | 4 | Live while dragging | P2 |
| Per-element styling (shape/color/font) | 4 (basic in 3.2.1 Pro) | Styling panel | P2 |
| Connection styling (line/decoration/routing) | 4 | Styling | P2 |
| Comment nodes (Markdown + images) | 3+4 (MD=4) | Toolbar | P2 |
| Higher-order constructs (4 approaches) | 3+4 | Canvas + two-stage | P1 |
| Moderating effect tool | 3+4 | Toolbar / right-click | P1 |
| Quadratic effect tool | 3+4 | Toolbar/menu | P2 |

### 2.3 Estimation / analyses
| Feature | Ver | UI Location | Priority |
|---|---|---|---|
| PLS-SEM Algorithm | 3+4 | Calculate menu | P0 |
| Bootstrapping | 3+4 | Calculate menu | P0 |
| Consistent PLS (PLSc, rho_A) | 3+4 | Calculate menu | P1 |
| Consistent Bootstrapping | 3+4 | Calculate menu | P1 |
| Blindfolding (Q²) | **3 only** | Calculate menu | P1 |
| PLSpredict (Q²predict, RMSE/MAE/MAPE, LM) | 3+4 | Calculate menu | P1 |
| CVPAT | **4 only** | Calculate menu | P2 |
| IPMA | 3+4 | Calculate menu | P1 |
| CTA-PLS | 3+4 | Calculate menu | P2 |
| Multigroup Analysis (PLS-MGA) | 3+4 | Calculate menu | P1 |
| Permutation + MICOM | 3+4 | Calculate menu | P1 |
| FIMIX-PLS | 3+4 | Calculate menu | P2 |
| PLS-POS | 3+4 | Calculate menu | P2 |
| Moderation (two-stage/product/ortho) | 3+4 | Modeling + algorithm | P1 |
| Quadratic / nonlinear | 3+4 | Modeling + algorithm | P2 |
| Mediation (specific/total indirect, VAF) | 3+4 | Bootstrap results | P1 |
| Model Fit (SRMR, d_ULS, d_G, NFI, χ², RMS_θ) | 3+4 | Quality criteria | P1 |
| CB-SEM + CFA | **4 only** | Calculate menu | P2 |
| CB-SEM Bootstrapping/MGA/MICOM/Moderation | **4 only** | Calculate menu | P2 |
| GSCA | **4 only** | Calculate menu | P2 |
| Regression / Logistic Regression | **4 only** | Regression menu | P2 |
| Path Analysis & PROCESS | **4 only** | Calculate menu | P2 |
| NCA + NCA Permutation | **4 only** | Calculate menu | P2 |
| Gaussian Copula endogeneity | **4 only** | Calculate menu | P2 |
| Model Comparison (BIC) | **4 only** | Calculate menu | P2 |

### 2.4 Reporting / UX
| Feature | Ver | UI Location | Priority |
|---|---|---|---|
| Results overlaid on model | 3+4 | Graphical view | P0 |
| Report tree (5 categories in v4) | 4 (deeper tree in 3) | Report View | P0 |
| Matrix / List / Bar chart per result | 3+4 | Report View | P0 |
| Color-coded threshold cells | 3+4 | Report tables | P0 |
| Toggle Zero Values | 3+4 | Report toolbar | P1 |
| Save / reopen reports | 4 | Report View | P1 |
| Report Comparison (Compare View) | 4 | Compare View | P1 |
| Export Excel / HTML / R / PDF / PNG / SVG | 3+4 (R≥3.1.3) | Report toolbar | P1 |
| Dark mode / color-blind themes | 4 | Preferences | P1 |
| Highlight paths (thickness by magnitude) | 3+4 | Graphical view | P2 |

---

## 3. Modeling / Canvas UX (detailed)

### 3.1 Modeling View layout
- **Left:** Indicators (manifest variables) list from the active data file. Supports type-ahead jump, a dynamic **Sort** icon (alphabetic / other), and a `+` collapsed-construct marker.
- **Right:** Modeling canvas (drawing board).
- **Top:** Main toolbar (modeling). **v4 also:** right-edge alignment toolbar.
- Indicators **already used** in the model render in **RED font** in the list; unused = **BLACK**.

### 3.2 Creating constructs
- **Drag-drop (recommended):** select one/several indicators (Ctrl-click to multi-select; click again to deselect) → drag to **empty canvas** → SmartPLS auto-creates a construct around them and pops a **name text field** (v4 hint: "press ENTER to confirm").
- **Drawing mode (classic):** toolbar Drawing mode → left-click empty canvas → empty latent variable with auto label `lv0, lv1, lv2…` → rename.
- Default shape = **ellipse/circle**. v4 also: rectangle, hexagon, octagon.
- New construct defaults to **reflective** (Mode A).

### 3.3 Attaching / detaching indicators
- Drag selected indicators **onto an existing construct** → added to its measurement model (arrow direction follows current mode: reflective = construct→indicator; formative = indicator→construct).
- Drop onto **empty canvas** → new construct.
- Indicators render as small rectangles with measurement arrows.
- An indicator **can be shared across constructs** (e.g., repeated-indicators HOC).
- **Single-indicator construct:** the arrow between the lone indicator and construct is removed (construct = indicator).
- v4 shows **scale-type markers** (metric/binary/ordinal/categorical).

### 3.4 Reflective ↔ formative
- Right-click construct → **Invert measurement model** (v4) / **Reverse Link** (v3, Ctrl+R) flips arrow direction.
- Estimation mode is **separate** from arrow direction. Double-click construct → choose outer weighting: **Mode A** (reflective default, correlation weights), **Mode B** (formative default, regression weights), **Equal Weights/Sumscores**, **Pre-Defined Weights** (≥ v3.2.2).

### 3.5 Higher-order constructs (HOC)
Four approaches to implement:
1. **Repeated indicators** — attach the LOC indicators (repeated) to the HOC; connect HOC↔LOCs.
2. **Extended repeated indicators** (total-effects, Becker et al. 2012) for collect-type HCMs.
3. **Embedded two-stage.**
4. **Disjoint two-stage** — Stage 1 estimates LOCs → **create data file of LOC latent-variable scores** (v4: "create data file from results") → use scores as HOC indicators in Stage 2.

Four HOC types: reflective-reflective, reflective-formative, formative-reflective, formative-formative (set per LOC/HOC via Mode A/B + Invert). Disjoint two-stage uses **standard settings** (Mode A reflective LOCs, Mode B formative LOCs) on both stages.

### 3.6 Effects on the canvas
- **Moderating effect (v4):** Moderating-effect tool → click moderator → drag onto the path to moderate; interaction term auto-generated. **v3:** right-click Y construct → Create Moderating Effect (wizard: moderator var, predictor var, interaction-generation method). Auto-adds the moderator→DV direct path at calculation time. Double-click to edit; tooltip shows settings.
- **Quadratic effect:** select Quadratic-effect option → click the target **path** (not construct). Two-stage squaring of saved scores. Double-click to edit.

### 3.7 Toolbar inventory + shortcuts (classic mapping; STRG=Ctrl)
| Shortcut | Action | Shortcut | Action |
|---|---|---|---|
| Ctrl+A | Select all | Ctrl+K | Align indicators below |
| Ctrl+B | Send to back | Ctrl+L | Align indicators right |
| Ctrl+C | **Calculate** (not copy) | Ctrl+N | New |
| Ctrl+D | Delete | Ctrl+O | Open |
| Ctrl+F | Bring to front | Ctrl+R | Reverse link |
| Ctrl+G | Group | Ctrl+S | Save |
| Ctrl+H | Hide indicators | Ctrl+T | Grid on/off |
| Ctrl+I | Align indicators above | Ctrl+U | Ungroup |
| Ctrl+J | Align indicators left | Ctrl+V | Show indicators |
| Ctrl+X | **Export image** (not paste) | Ctrl+Y | Redo |
| | | Ctrl+Z | Undo |

> ⚠️ Recommend remapping Ctrl+C/Ctrl+X/Ctrl+V to standard clipboard semantics in the Python port (the classic mapping is confusing); expose Calculate via the wheel/gear icon and a clear shortcut (e.g., F5).

---

## 4. Canvas Interaction Rules (exact expected behaviors)

> These are literal acceptance criteria — replicate exactly.

1. **Delete construct ⇒ deletes its attached indicator boxes + measurement arrows on the canvas, but NEVER deletes underlying data columns.** Freed indicators revert **red → black** in the left list (available to reuse).
2. **Delete indicator ⇒ removes only that indicator box from the construct; construct remains; data column untouched; indicator frees in list.**
3. **Delete connection ⇒ removes only that structural path.**
4. **Removing all indicators from a construct ⇒ leaves an EMPTY (invalid/red) construct, not a clean one.**
5. **Drag construct ⇒ moves the whole measurement block (construct + indicators + arrows) as a unit.** Indicators can also be moved independently.
6. **Shift+drag ⇒ constrain movement to horizontal or vertical.**
7. **Ctrl/Cmd+drag ⇒ pans the canvas (v4), does NOT move elements.**
8. **Mouse wheel ⇒ zoom in/out.** Original-size resets to 100%. Zoom level persists across reload (v4).
9. **Right-click empty canvas ⇒ reverts to Selection mode (classic).**
10. **Double-click construct ⇒ SETTINGS dialog** (name + measurement/weighting mode). **Right-click ⇒ CONTEXT MENU.**
11. **Connections are construct-to-construct ONLY.** v4: cannot manually draw connections to/from manifest indicators. Classic Connection mode shows center **ports** for port-to-port dragging.
12. **Hide indicators ⇒ collapse all indicator boxes; construct LABEL becomes BOLD; v4 adds a `+` marker.** Show indicators reverses.
13. **Group (Ctrl+G) / Ungroup (Ctrl+U)** bind arbitrary elements to move together; **Bring to front / Send to back** manage overlap.
14. **ALT+SHIFT+click+drag a construct ⇒ interactively align its indicators.**

### 4.1 Status coloring (compute automatically)
| Visual | Meaning |
|---|---|
| **RED construct** on canvas | Invalid/incomplete: no indicators assigned, or not connected into the structural model. Hover ⇒ tooltip stating the exact problem. |
| **BLUE construct** on canvas | Valid/complete: has indicators and is properly connected. |
| **RED font** in left list | That indicator is **already used** in the model. |
| **BLACK font** in left list | Indicator is available/unused. |

> ⚠️ Two distinct reds — do not conflate: red *construct* = invalid; red *list font* = used.
> **Calculate is blocked while any element is red/invalid;** show an error message + highlight the offending element.

### 4.2 Context menu — construct (full item list)
`Invert measurement model` / `Reverse Link` · `Align/Arrange Indicators` (Above Ctrl+I / Below Ctrl+K / Left Ctrl+J / Right Ctrl+L) · `Hide Indicators` (Ctrl+H) / `Show Indicators` (Ctrl+V) · `Create Moderating Effect` · `Add Quadratic Effect` · `Rename Object` · `Delete` · `Cut/Copy/Paste` · `Group/Ungroup` · `Bring to front / Send to back` · styling (colors/borders/font/shape).

### 4.3 Context menu — indicator (limited)
`Delete` (frees in list; data untouched) · selection/copy · styling where applicable. Arrow direction is controlled at the **construct** level (Invert), not per-indicator. Hide/Show is a construct-level operation.

---

## 5. PLS-SEM Algorithm + Results Reports

### 5.1 Algorithm setup dialog
Tabbed: **Setup** + **Data**, plus an `Open report` checkbox and a `Start calculation` button.

**Setup tab**
| Option | v3 | v4 | Notes |
|---|---|---|---|
| Weighting scheme | Path (default) / Factor / **Centroid** | Path (default) / Factor / **PCA** | Path maximizes R² of endogenous LVs; Factor differs little. Centroid removed in v4; PCA added. |
| Type of results / Data metric | Standardized (Mean 0, Var 1) default; Unstandardized/Original | Standardized (default) / Unstandardized / Mean-centered | Unstandardized only if all indicators share scale. |
| Initial weight | 1.0 default; Individual list | 1.0 default; Individual (Min box + "Apply for all indicators" + Reset) | Rarely changed. |
| Max iterations | **Editable, default 300** | **Fixed 3,000** | KEY DIFFERENCE. |
| Stop criterion | **Editable**, exponent X in 10⁻ˣ (rec. 7) | **Fixed 1.0E-7** | KEY DIFFERENCE. |

**Data tab**
- Missing value algorithm: None / Mean value replacement / Casewise deletion / Pairwise deletion. (Missing marker set at import.)
- Weighting vector: None (default) or a weighting variable.
- Data groups: select groups for group-specific estimation / MGA.

> **Singular data matrix error:** raised at start if an indicator is constant (zero variance) or duplicated/linearly dependent — must be removed.

### 5.2 Graphical output (on the model)
- Path coefficients on structural arrows; **R² inside endogenous construct circles**; outer **loadings (reflective)** / **weights (formative)** on indicator arrows.
- Top control strip: **Inner model** selector, **Outer model** selector, **Constructs** selector cycling **AVE / Composite reliability (rho_a) / Composite reliability (rho_c) / Cronbach's alpha / R-square / R-square adjusted** (click then ↑/↓ to cycle).
- `Highlight paths` checkbox ⇒ arrow thickness ∝ coefficient magnitude.

### 5.3 Report tree
- **v4 — 5 top categories (exact order):** `1) Graphical · 2) Final results · 3) Quality criteria · 4) Algorithm · 5) Model and data`.
- **v3 — deeper path style:** `Report > Default Report > PLS > Calculation Results > …` and `… > Quality Criteria > …`.

### 5.4 Final results
| Result | Columns / notes | Thresholds |
|---|---|---|
| Path Coefficients | Matrix (row→column) / List / Bar chart | -1..+1; \|coef\|>1 ⇒ collinearity. Rule of thumb (n≤~1000): \|path\|>0.20 likely sig, <0.10 likely not — confirm via bootstrap. |
| Indirect Effects / Specific indirect / Total indirect | Matrix/List | For mediation; significance via bootstrap. |
| Total Effects | Matrix/List | Feeds IPMA. (v3 grouped under Quality Criteria; v4 under Final results.) |
| Outer Loadings | per indicator-construct | Reflective: ≥0.708 (green ≥0.7, red <0.7); loading² = indicator reliability. |
| Outer Weights | per indicator-construct | Formative: interpret significance + relevance via bootstrap. |
| Latent Variable Scores | per case | Standardized; "Index values" also available; feeds VIF/IPMA. |
| Latent Variable Correlations (LVC) | matrix | Feeds Fornell-Larcker. |
| Residuals | residual correlation matrices | Diagnostic. |

### 5.5 Quality criteria
| Result | Columns | Color thresholds |
|---|---|---|
| R-square / R-square adjusted | per endogenous LV | 0.25 / 0.50 / 0.75 = weak / moderate / substantial. |
| f-square | matrix | Cohen 0.02/0.15/0.35. Color: green ≥0.15, black ≥0.02, red <0.02. |
| Construct Reliability & Validity | Cronbach's α, rho_a, rho_c, AVE | Reliability green ≥0.7 / red <0.7 (0.70–0.90 good; >0.95 redundant). AVE green ≥0.5 / red <0.5. |
| Discriminant — Fornell-Larcker | sqrt(AVE) diagonal vs correlations | sqrt(AVE) must exceed inter-construct correlations. |
| Discriminant — HTMT | matrix (absolute correlations) | green ≤0.85, black ≤0.90, red >0.90. Use 0.85 (distinct) / 0.90 (similar). Inference via bootstrap CI. |
| Discriminant — Cross loadings | matrix | Own-construct loading must be highest. |
| Collinearity (VIF) — Outer / Inner | per indicator / per predictor | green ≤3, black ≤5, red >5. |
| Model Fit | see §10 | SRMR <0.08/<0.10; NFI >0.9; RMS_θ <0.12; d_ULS/d_G via bootstrap CI. |

### 5.6 Algorithm / Model and data
- `Stop criterion changes` (iteration log — confirm convergence before max iterations).
- Settings echo; processed/index data matrices.

---

## 6. Bootstrapping (separate report)

### 6.1 Setup (tabbed: Data + PLS setup carry over; **BT setup** is the new tab)
| Option | Values / default | Notes |
|---|---|---|
| Subsamples | 500 (quick) / 5,000 (interim) / **10,000 (final)** | Each subsample = N obs, drawn with replacement. |
| Do parallel processing | checkbox, recommend **ON** | |
| Amount of results | **Most important (faster)** vs **Complete (slower)** | v3 wording: Basic vs Complete. HTMT/R²/reliability inference ⇒ Complete required. |
| Confidence interval method | **Percentile (default)** / Studentized / **BCa** | BCa for non-normal/asymmetric distributions. |
| Test type | One-tailed / **Two-tailed** | HTMT DV workflow uses one-tailed. |
| Significance level | 0.01 / **0.05** / 0.10 | 0.05 ⇒ 95% CI. |
| Random number generator | **Random seed (v3 only)** / **Fixed seed (v4)** | Fixed seed for reproducibility. |

> **Official primer workflow:** retain prior settings, parallel ON, 10,000 subsamples, Complete, Percentile, Two-tailed, 0.05, Fixed seed, Open report, Start.

### 6.2 Result tabs (per quantity)
| Tab | Columns | Meaning |
|---|---|---|
| Mean, STDEV, T values, P values | `Original Sample (O)`, `Sample Mean (M)`, `Standard Deviation (STDEV)` (v3: `STERR`), `T statistics (\|O/STDEV\|)`, `P values` | **Report O** (not M). O−M gap = bias/skew. STDEV = bootstrap SE = t denominator. |
| Confidence Intervals | O, M, `Bias`, percentile bounds (e.g. 2.5% / 97.5%) | **Plain percentile, NOT bias-corrected.** Bias = M − O. |
| Confidence Intervals Bias Corrected | O, M, Bias, corrected bounds | BCa if selected, else bias-corrected percentile. **Report this for bias-corrected CIs.** |
| Samples | one row per subsample | Raw empirical distribution; feeds histograms. |
| Histograms | frequency plot | Judge normality ⇒ percentile vs BCa. |

> ⚠️ **UX trap:** selecting BCa in setup is not enough — you must read the **Confidence Intervals Bias Corrected** tab; the plain `Confidence Intervals` tab is always percentile.

### 6.3 Significance rules (three equivalent)
- \|T\| > critical: two-tailed **1.65 / 1.96 / 2.57** (10/5/1%); one-tailed **1.28 / 1.65 / 2.33**.
- P < significance level (0.05) — color green ≤0.05 / red >0.05.
- CI (preferably bias-corrected) excludes zero.
- When they disagree at the margin, primer recommends the bias-corrected bootstrap CI.

### 6.4 Variants
- **Consistent Bootstrapping** — for PLSc models (reflective common-factor). Same dialog/report layout.
- **v4 only:** CB-SEM Bootstrapping, Regression Bootstrapping, Path Analysis/PROCESS Bootstrapping — same BT setup + Original/Mean/STDEV/T/P + CI tabs.

---

## 7. Advanced predictive / prioritization

### 7.1 Blindfolding — Q² (SmartPLS 3 only)
- **Setup:** Omission distance **D** (default 7) + inherited PLS settings. **Constraint: N/D must NOT be an integer.** Recommended D ∈ [5,12].
- **Results:** `Construct Crossvalidated Redundancy` and `Construct Crossvalidated Communality` — columns `SSO`, `SSE`, `Q² (=1-SSE/SSO)`; per-indicator variants in full report.
- **Thresholds:** Q² > 0 ⇒ predictive relevance (use **redundancy** variant). Magnitude 0.25 medium / 0.50 large. q² effect size = (Q²_incl − Q²_excl)/(1 − Q²_incl) at 0.02/0.15/0.35.
- **v4:** removed; use PLSpredict + CVPAT. (Doc index page still lists Blindfolding but it is non-runnable.)

### 7.2 PLSpredict — out-of-sample Q²predict (3 + 4)
- **Setup:** `Number of folds` k (default **10**), `Number of repetitions` (default **10**); PLS algorithm sub-settings.
- **Results:**
  - **LV Prediction Summary:** `Q²predict`, `RMSE`, `MAE` (no LM, no MAPE at LV level).
  - **MV Prediction Summary:** `Q²predict`, `PLS-SEM_RMSE/MAE/MAPE`, `LM_RMSE/MAE/MAPE`, `PLS-SEM − LM_RMSE`, `PLS-SEM − LM_MAE`.
  - Per-case Predictions / Prediction Errors.
- **Decision:** (1) Q²predict > 0 beats naive indicator-average benchmark. (2) symmetric errors ⇒ RMSE, else MAE. (3) PLS vs LM across target indicators ⇒ **High** (all lower) / **Medium** (majority) / **Low** (minority) / **None** (none).
- **Color:** Q²predict green ≥0 / red <0; PLS-SEM_RMSE green if < LM_RMSE; PLS-SEM_MAE green if < LM_MAE.

### 7.3 CVPAT (v4 only)
- Setup: folds (10), repetitions (10), benchmark IA and/or LM.
- Result: **average loss difference** vs IA and vs LM, with t/p and CIs.
- Rule: significantly **negative** loss difference ⇒ model predicts better than benchmark.

### 7.4 IPMA (3 + 4)
- **Setup (3 areas):** (1) **Target construct** dropdown. (2) **IPMA Results** display: "All Predecessors … (Including MV Charts)" or "Direct Predecessors … (Including MV Charts)". (3) **Ranges** grid — verify/correct each MV's theoretical min/max (e.g., 7-point Likert ⇒ 1..7). Getting Ranges wrong silently distorts performance.
- **Importance** = unstandardized **total effect** on target. **Performance** = mean of rescaled LV scores (0–100).
- **Output:** Importance-Performance Map (x=importance, y=performance) with vertical mean-importance + horizontal mean-performance lines forming 4 quadrants; high-importance/low-performance = priority. Construct- and indicator-level tables.
- **Requirements:** metric/quasi-metric indicators, all coded same direction (recode reverse items first).

---

## 8. Multi-group & Segmentation

### 8.1 Data Groups (prerequisite)
Create in Data View **before** MGA/Permutation/MICOM: **Add Data Group** (name + filter, e.g. `gender == 1`) or **Generate Data Groups** (one group per category value). Each group ≥ minimum sample (≈10× max arrows into any construct, or 10× largest formative-indicator count). Shared `Select Groups` panel: every group in Group A is compared vs every group in Group B.

### 8.2 Multigroup Analysis (PLS-MGA)
- Built from group-specific bootstrapping (+ permutation). Reports **Confidence-interval overlap**, **PLS-MGA (Henseler)**, **Parametric** (equal variances), **Welch-Satterthwaite** (unequal variances).
- Setup: Select Groups + PLS/bootstrap settings (weighting, subsamples ~5000, CI method, sig level, tails, parallel, sign-change handling).
- Columns: group-specific estimates (A, B), `diff (\|A−B\|)`, `p-value (one-tailed)`, `p-value (two-tailed)`, PLS-MGA p, bias-corrected CIs per group.
- **Decision:** PLS-MGA significant if p<0.05 OR p>0.95; parametric/Welch significant if p<0.05; CI test significant if bias-corrected CIs do NOT overlap. Requires **MICOM partial invariance first**.

### 8.3 Permutation + MICOM
- **Permutation** (reassigns obs without replacement, constant group sizes): permutations default 1000 (5000 final), tails, sig level, parallel. Produces **both** permutation-MGA tables and full MICOM tables in one report.
- **MICOM 3 steps:**
  - Step 1 Configural (qualitative checklist; identical setup).
  - Step 2 Compositional: Original correlation `c`, Permutation mean, 5% quantile, p-value. Established if c **not significantly < 1** (p>0.05).
  - Step 3 Equal means & variances: Original difference, 2.5%/97.5% CI, p-value. Established if differences fall within CI (p>0.05). **Correction:** CI should include the obtained difference (not zero).
  - **1+2 = partial invariance** (sufficient for MGA path comparison); **1+2+3 = full invariance** (may pool data).

### 8.4 CB-SEM MGA (v4 only)
Constraint-based (Fixed values / Equality / Across-group constraints); nested-model invariance (configural → metric → scalar) via chi-square difference. No PLS-MGA resampling logic.

### 8.5 FIMIX-PLS
- Latent-class (EM) segmentation of the **structural** model (reflective).
- **Setup:** Number of segments (run for 1,2,3…), Max iterations, Stop criterion (EM ΔLnL), Number of repetitions (restarts), optional unstandardized scores / segment intercepts.
- **Fit indices:** AIC, AIC3, AIC4, BIC, CAIC, HQ, MDL5, LnL, EN (entropy), NFI, NEC; + relative segment sizes, segment path coefficients, per-obs membership probabilities.
- **Decide #segments:** combine criteria — when **AIC3 and CAIC agree**, that count; AIC4+BIC as a pair; pick fewer than AIC suggests, more than MDL5. EN > 0.50 = clean separation. Each segment ≥ 10× rule.

### 8.6 PLS-POS
- Distance-based (hill-climbing) segmentation of **structural + measurement** models (reflective & formative).
- **Setup:** Number of segments, Max iterations, **Search Depth** (set = N in final runs), **Initial Separation** (random or FIMIX-based), optional **Pre-Segmentation**, **Optimization Criterion** (sum all-construct R² / target R² / weighted variants), Target Construct.
- No information criteria; judge by R² improvement + interpretability; run with different starts. Validate via MICOM + MGA.

> **Downstream loop:** segmentation → read per-obs membership → turn into data group → MICOM + MGA on segments → characterize segments with an explanatory variable.

---

## 9. Moderation / Nonlinear / Mediation / NCA

### 9.1 Moderation
- **v3 creation:** select dependent construct → toolbar Moderating-effect OR right-click → Create Moderating Effect → wizard (Moderator, Predictor, generation method) → Finish.
- **v4 creation:** draw path from moderator onto existing structural path → double-click the moderation path → interaction dialog auto-generates predictor×moderator term. Supports **three-way / multiple moderation**.
- **Methods:** **Two-Stage** (default, recommended; works with reflective/formative/single-item) · **Product Indicator** (reflective only; Mean-Centering / Double Mean-Centering = CB-SEM default / Generic) · **Orthogonalizing** (least bias, max explained variance).
- Data metric (Standardized/Unstandardized/Mean-centered) inherited from algorithm dialog.
- **Outputs:** interaction path coefficient + t/p/CI; **f² interaction** (Kenny: ~0.005 small / 0.01 medium / 0.025 large — smaller than Cohen); **Simple Slope Plot** (lines at −1 SD / mean / +1 SD; green=+1 SD, red=−1 SD; x=predictor, y=outcome).

### 9.2 Quadratic / nonlinear
- Select Quadratic-effect option → click target path. Two-stage: save predictor LV scores, square them.
- Model y = b₁X + b₂X². **Negative b₂ ⇒ inverted-U; positive b₂ ⇒ U.** Significant quadratic often coexists with non-significant linear.

### 9.3 Mediation
- Build mediator(s) → run PLS-SEM + Bootstrapping.
- Tables: `Direct Effects`, `Specific Indirect Effects`, `Total Indirect Effects`, `Total Effects` — each with O, M, STDEV, t, p, CI.
- **Zhao/Hair typology:** indirect sig + direct sig same sign = complementary (partial); indirect sig + direct sig opposite = competitive (suppression); indirect sig + direct n.s. = indirect-only (full); indirect n.s. + direct sig = direct-only (no mediation); both n.s. = no effect.
- **VAF** = indirect / total (computed manually, NOT in UI): <20% none · 20–80% partial · >80% full. Breaks down with suppression — prefer bootstrapped indirect-effect significance.

### 9.4 Path Analysis & PROCESS (v4 beta)
Regression-based, one-step, equally weighted indicators on unstandardized data; supports moderated mediation with bootstrapped conditional direct/indirect effects. Options: Data metric (Unstandardized/Mean-centered/Standardized), Control variables.

### 9.5 NCA (v4 only)
- **Algorithms:** `Necessary Condition Analysis (NCA)` + `NCA Permutation` (+ combined cIPMA).
- **Setup:** NCA — `Number of steps for bottleneck tables` (default 10 = 10% steps). Permutation — subsamples (≥1000 init, 5000+ final), parallel, sig level, random/fixed seed.
- **Outputs:** scatter with **CE-FDH** (step function, 100% accurate) and **CR-FDH** (regression, ≤100%) ceiling lines; effect size **d** + ceiling accuracy; **Bottleneck table** (required min X per Y level, actual/percent/percentile); permutation p for d.
- **Thresholds (Dul 2016):** 0<d<0.1 small · 0.1–0.3 medium · 0.3–0.5 large · ≥0.5 very large. Report if **d≥0.1 AND p<0.05**. Accuracy benchmark ~95%. NCA-ESSE for extreme-response sensitivity.

---

## 10. Consistent PLS / CTA / Model Fit

### 10.1 Consistent PLS (PLSc)
- Disattenuates correlations between reflective LVs to mimic CB-SEM (correction-for-attenuation). Introduces **rho_A** (between Cronbach's α and rho_c) — the only consistent reliability for PLS scores.
- Setup ≈ PLS dialog (Path/Factor/Centroid, Max iter 300, stop 10⁻⁷, init 1.0); option to connect all LVs when generating scores (Dijkstra & Henseler advise connecting all).
- rho_A ≥ 0.70. Can return `n/a`/non-convergence with very low reliabilities/misspecified factor model.
- Pair with **Consistent Bootstrapping** for inference.

### 10.2 CTA-PLS (reflective vs formative test)
- Tests model-implied **vanishing tetrads**. Bootstrapping (500 init → 10,000 final), tails, sig level. Each construct needs **≥4 indicators** (≤~25).
- Per-construct: tetrad value, bias, SE, t, p, bias-corrected CI + **Bonferroni-adjusted** CI bounds (CI Low adj. / CI Up adj.).
- **Decision:** adjusted bias-corrected CI **includes 0** ⇒ vanishes ⇒ **reflective**; **excludes 0** (sig) for any non-redundant tetrad ⇒ **formative**.

### 10.3 Model Fit
Reported for **Saturated** vs **Estimated** model.
| Measure | Threshold |
|---|---|
| SRMR | <0.08 conservative; <0.10 lenient |
| d_ULS (squared Euclidean) | No standalone cutoff — original must be **below** bootstrap HI95/HI99 upper bound (exact fit not rejected, p>0.05) |
| d_G (geodesic) | Same as d_ULS (bootstrap CI) |
| Chi-square | (N−1)·ML; df = (K²+K)/2 − t |
| NFI | >0.90 acceptable (no complexity penalty) |
| RMS_theta | <0.12 good (reflective outer model only) |

> SmartPLS docs caution PLS-SEM fit indices are immature; recommend reporting the **Estimated** model and not relying on GoF.

---

## 11. UI/UX Visual Design

### 11.1 Shell / views
- Two regions: top **Menu+Toolbar band**, bottom **Working Area** = **left navigation tree + right content panel** that re-skins per view.
- **5 views:** Project · Data · Modeling · Report · **Report Comparison (v4)**.
- v3 = light-only Java/Swing (gray panels, MDI windows via `Window` menu, plain ovals/thin arrows). v4 = single harmonized shell, denser/cleaner, styling toolbar, dark + color-blind themes, recolored icons, removed text shadows.

### 11.2 Menus
- **v4:** `Files` (New project, Import from backup) · `Edit` · `View` · `Calculate` (PLS-SEM, Bootstrapping, IPMA, …) · `Regression` · `Help`.
- **v3:** `File · Edit · View · PLS (calculate) · Window · Help`.
- Calculate also via the **wheel/gear "Calculate" icon** (must be in Modeling window for it to be active).

### 11.3 Toolbar icons (modeling)
Selection · Drawing · Connection · **Calculate (wheel/gear)** · Delete · Grid on/off · Zoom in · Zoom out · Original size · Undo · Redo · Group · Ungroup · Bring to front · Send to back · Comment · Moderating effect · Quadratic effect · (v4) styling + alignment sections.

### 11.4 Results display
- Each numeric result: **Matrix (row→column)** / **List** / **Bar chart** (value labels under bars). `Toggle Zero Values` icon; adjustable decimals; chart info bar with customizable X/Y tick units + line types (v4).
- Graphical view paints estimates on the model (loadings/weights on outer arrows, paths on inner, **R² in circles**).

### 11.5 Color thresholds (replicate exactly)
| Metric | Green | Black | Red |
|---|---|---|---|
| Outer loadings | ≥0.70 | — | <0.70 |
| Cronbach's α / rho_a / rho_c | ≥0.70 | — | <0.70 |
| AVE | ≥0.50 | — | <0.50 |
| HTMT | ≤0.85 | ≤0.90 | >0.90 |
| VIF | ≤3 | ≤5 | >5 |
| f-square | ≥0.15 | ≥0.02 | <0.02 |
| p-values | ≤0.05 | — | >0.05 |
| Q²predict | ≥0 | — | <0 |
| PLS-SEM RMSE/MAE vs LM | < LM (green) | — | ≥ LM (red) |

### 11.6 Theming (v4)
- Themes: **Normal (light) · Dark · Color-blind** (also true-white black-and-white #ffffff). Configurable default font. Toolbar icons tuned for dark-mode.
- **Image/graphic export defaults to a LIGHT background even in dark mode**, with a custom background-color override; pasted Office tables are numeric, no black background.

### 11.7 Export
Save to project (reopen later) · Excel `.xlsx` (numeric cells, >256 cols) · HTML/web · R (full report or single matrix, ≥3.1.3) · PDF · PNG · SVG · copy tables to Excel/Word.

---

## 12. UI/UX Polish Recommendations (match or exceed SmartPLS 4)

1. **Target v4 semantics:** rho_a/rho_c naming, fixed convergence (3,000 / 1.0E-7), Path/Factor/PCA, Compare View, Fixed-seed bootstrap, STDEV labeling.
2. **Fix the legacy shortcut traps:** make Ctrl+C/X/V standard clipboard; bind Calculate to the wheel icon + F5; keep the align/group/hide set.
3. **First-class status feedback:** live red/blue construct validity with hover tooltips stating the *exact* missing condition; block Calculate with an inline highlight, not just a modal.
4. **Snappy canvas:** GPU-accelerated `QGraphicsScene/View`; wheel-zoom + Ctrl/Cmd-drag pan; Shift-constrained drag; grid + snap; **green-ruler alignment guides** that also snap to construct/connection labels; "reset all label positions" action.
5. **Modern styling system:** per-element shape (circle/rectangle/hexagon/octagon), fill/border/font, connection line styles + decorations + corner routing; saveable style presets/themes; **Apply Theme undoable**.
6. **Comment nodes with Markdown + embedded images** (headings, bold/italic, lists, links, `![](image:...)`).
7. **Robust import:** native CSV/Excel/SPSS; explain *why* the Import button is disabled; descriptives + scale-type detection in Data View.
8. **Reports that scale:** saveable/reopenable, side-by-side Compare View, virtualized matrix tables, Matrix/List/Bar toggle, Toggle Zero Values, decimal control, threshold coloring everywhere, export to Excel/HTML/R/PDF/PNG/SVG.
9. **Reproducibility + speed:** parallel bootstrap (multiprocessing/numpy/numba), fixed-seed RNG, progress with cancel.
10. **Two-stage HOC convenience:** "Create data file from results" to feed Stage 2 directly (eliminates manual score export).
11. **Accessibility:** Dark + color-blind themes; keyboard navigation; light-background image export regardless of theme; remove text shadows for B/W readability.
12. **Exceed SmartPLS:** add an integrated **assumptions/diagnostics panel** (constant-indicator, collinearity, convergence) before Calculate; an inline **reporting helper** that drafts APA-style result sentences from the thresholds; one-click **PLS-SEM ↔ PLSc ↔ CB-SEM** triangulation in Compare View; scriptable Python API mirroring the GUI.

---

## 13. References

**SmartPLS documentation & tutorials**
- https://www.smartpls.com/documentation/tutorials/first-pls-path-model/
- https://smartpls.com/documentation/tutorials/first-cb-sem-model/
- https://smartpls.com/documentation/tutorials/first-project/
- https://smartpls.com/documentation/tutorials/first-steps/
- https://www.smartpls.com/documentation/tutorials/import-datafiles/
- https://www.smartpls.com/release_notes/
- https://www.smartpls.com/faq/smartpls4/whatsnew/
- https://www.smartpls.com/faq/smartpls4/commenting-models/
- https://www.smartpls.com/faq/smartpls4/plsc-problems/

**Algorithms & techniques**
- https://www.smartpls.com/documentation/algorithms-and-techniques/pls/
- https://www.smartpls.com/documentation/algorithms-and-techniques/bootstrapping/
- https://www.smartpls.com/documentation/algorithms-and-techniques/cbsem-bootstrapping/
- https://www.smartpls.com/documentation/algorithms-and-techniques/consistent-bootstrapping/
- https://www.smartpls.com/documentation/algorithms-and-techniques/path-analysis-and-process-bootstrapping/
- https://www.smartpls.com/documentation/algorithms-and-techniques/regression-bootstrapping/
- https://www.smartpls.com/documentation/algorithms-and-techniques/blindfolding/
- https://smartpls.com/documentation/algorithms-and-techniques/predict/
- https://smartpls.com/documentation/algorithms-and-techniques/cvpat/
- https://www.smartpls.com/documentation/algorithms-and-techniques/ipma/
- https://www.smartpls.com/documentation/algorithms-and-techniques/multigroup-analysis/
- https://smartpls.com/documentation/algorithms-and-techniques/cbsem-multigroup-analysis/
- https://smartpls.com/documentation/algorithms-and-techniques/permutation/
- https://www.smartpls.com/documentation/algorithms-and-techniques/micom/
- https://www.smartpls.com/documentation/algorithms-and-techniques/fimix-pls/
- https://smartpls.com/documentation/algorithms-and-techniques/pos/
- https://www.smartpls.com/documentation/algorithms-and-techniques/moderation/
- https://smartpls.com/documentation/algorithms-and-techniques/cbsem-moderation/
- https://www.smartpls.com/documentation/algorithms-and-techniques/nonlinear/
- https://www.smartpls.com/documentation/algorithms-and-techniques/mediation/
- https://www.smartpls.com/documentation/algorithms-and-techniques/nca/
- https://www.smartpls.com/documentation/algorithms-and-techniques/path-analysis-and-process/
- https://www.smartpls.com/documentation/algorithms-and-techniques/higher-order/
- https://www.smartpls.com/documentation/algorithms-and-techniques/consistent-pls/
- https://smartpls.com/documentation/algorithms-and-techniques/cta-pls/
- https://www.smartpls.com/documentation/algorithms-and-techniques/model-fit/
- https://www.smartpls.com/documentation/algorithms-and-techniques/discriminant-validity-assessment/
- https://www.smartpls.com/documentation/algorithms-and-techniques/prediction-oriented-model-selection/
- https://smartpls.com/documentation/algorithms-and-techniques/

**Functionalities, glossary, sample projects**
- https://www.smartpls.com/documentation/functionalities/thresholds/
- https://www.smartpls.com/documentation/functionalities/outer_weights_prespecification/
- https://www.smartpls.com/documentation/literature/glossary/
- https://www.smartpls.com/documentation/sample-projects/nca-corporate-reputation/
- https://www.smartpls.com/documentation/sample-projects/nca-tam/

**Primer case studies (Hair et al., A Primer on PLS-SEM, 3rd ed.)**
- https://www.smartpls.com/primer-book-case-studies/primer_3e_chap3_case_new.pdf
- https://www.smartpls.com/primer-book-case-studies/primer_3e_chap4_case_new.pdf
- https://www.smartpls.com/primer-book-case-studies/primer_3e_chap5_case_new.pdf
- https://www.smartpls.com/primer-book-case-studies/primer_3e_chap6_case_new.pdf
- https://www.smartpls.com/primer-book-case-studies/primer_3e_chap7_case_new.pdf

**Forum threads**
- https://forum.smartpls.com/viewtopic.php?t=16721
- https://forum.smartpls.com/viewtopic.php?t=16116
- https://forum.smartpls.com/viewtopic.php?t=961
- https://forum.smartpls.com/viewtopic.php?t=2173
- https://forum.smartpls.com/viewtopic.php?t=1406
- https://forum.smartpls.com/viewtopic.php?t=336
- https://forum.smartpls.com/viewtopic.php?t=3494
- https://forum.smartpls.com/viewtopic.php?t=1398
- https://forum.smartpls.com/viewtopic.php?t=3332
- https://forum.smartpls.com/viewtopic.php?t=3757
- https://forum.smartpls.com/viewtopic.php?t=30544
- https://forum.smartpls.com/viewtopic.php?t=30618

**Third-party / academic**
- http://somphdclub.weebly.com/uploads/2/3/8/5/23854145/3_smartpls_user_guide.pdf
- https://eli.johogo.com/Class/CCU/SEM/_Mastering-Partial-Least-Squares_Wong.pdf
- https://en.wikipedia.org/wiki/SmartPLS
- https://link.springer.com/article/10.1057/s41270-023-00266-y
- https://researchwithfawad.com/index.php/predictive-power-assessment-using-plspredict-in-smartpls3/
- https://researchwithfawad.com/index.php/lp-courses/smartpls4-tutorial-series/a-basic-and-simple-model-in-smartpls4/

**Key literature referenced by SmartPLS**
- Hair, Hult, Ringle & Sarstedt (2022) *A Primer on PLS-SEM*, 3rd ed.
- Hair, Risher, Sarstedt & Ringle (2019) *When to use and how to report the results of PLS-SEM*, EBR.
- Hair, Sarstedt, Ringle & Gudergan (2024) *Advanced Issues in PLS-SEM*, 2nd ed.
- Dijkstra & Henseler (2015) *Consistent PLS path modeling* (PLSc, rho_A).
- Henseler, Ringle & Sarstedt (2015) HTMT criterion.
- Gudergan, Ringle, Wende & Will (2008) CTA-PLS.
- Liengaard et al. (2021) CVPAT; Shmueli et al. (2019) PLSpredict.
- Dul (2016) Necessary Condition Analysis (NCA).
