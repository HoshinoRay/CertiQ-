# Grow-from-Seed 认证不变集扩张
### post-hoc certification of a learned robust Q-CBF artifact 的构造引擎 — 原理、定理、证明与实验技术框架

> **本文档定位。** 这不是一条新主线,而是主线的**构造引擎**。主线 claim 保持不变:
> *对一个冻结的、未约束的学习鲁棒 Q-CBF 三元组 `(V_θ, Q_θ, π♭)` 做 sound 的后验认证*。
> 本模块给出:如何从已认证的刹车种子 `S_brake` 出发,**构造性地、anytime-sound 地**把认证集长大到
> 接近最大鲁棒控制不变集,同时(i)认证集始终锚定在学习对象 `V_θ` 的零水平集内,(ii)`Q_θ`
> 作为运行时谓词被"限制到 verifier 认可的动作"后部署,(iii)未限制的 `Q_θ` 滤波器被**同一个
> verifier 拒绝**(false feasibility,Theorem S)。
>
> **关键结构性质(对应"通用性"提醒)。** 全部 soundness 证明**只依赖一个抽象接口 A3**(可靠包络
> `Enc ⊇ 真实可达集`),**从不依赖 `f` 的解析形式**。因此 toy(Dubins/F1TENTH,`f` 解析)只是
> 该接口的最强 instantiation;高维、`f` 不可解析建模的系统用"学习动力学 `f̂` + 可靠/PAC 误差包络"
> instantiate 同一接口,定理逐字成立(soundness 退化为分级 soundness,见 §7)。

---

## 1. 记号、设定与信任锚

### 1.1 系统与对手(A1)
离散时间受扰系统
$$x_{t+1} = f(x_t, u_t, d_t),\qquad u_t\in U,\; d_t\in D,$$
`f` 已知(或在 §7 中由学习模型 + 可靠误差代替),`D` 紧。对手 `d_t` **对抗地、带全状态反馈**选取(最强 discriminating 设定)。证书针对采样模型 + ZOH 输入;inter-sample 安全或显式 out-of-scope,或用 `K` 收缩 margin `sup‖ẋ‖·T` 处理(与现有 `S_brake` 口径一致)。

### 1.2 学习对象(被认证的主体)
冻结的 `(V_θ, Q_θ, π♭)`(高维时再加冻结的 `f̂`),plain MLP,**无 hard-Lipschitz、无 analytic D 嵌入**。`V_θ` 给出 membership backbone,`Q_θ` 给出运行时动作谓词
$$\Phi_\theta(x)=\{u:\ \min_{d}Q_\theta(x,u,d)\ge \gamma V_\theta(x)+\varepsilon\},$$
`π♭` 给出 fallback。**这三者除"提议/选择/度量"外永不进入安全推理链**(§5)。

### 1.3 种子(A2)
$$R_0 = S_{\mathrm{brake}}.$$
对每个 cell,已验证 `π♭` 的整条 brake-to-stop 轨迹在 `∀d` 下 `g≥0 (∧ V_θ≥0)`,终点 `v=0` 是鲁棒不动点。即 `R_0` 中每个状态拥有**构造性的永久安全策略**。这是已交付物。

### 1.4 抽象接口 / 可靠包络(A3 — 唯一的动力学信任锚)
存在可计算映射 `Enc`,对每个 cell `c` 与动作 `u`:
$$\mathrm{Enc}(c,u)\ \supseteq\ \mathrm{Reach}(c,u):=\{f(x,u,d):x\in c,\ d\in D\},$$
且包络单调收紧:`c'⊆c ⇒ Enc(c',u)⊆Enc(c,u)`,overshoot `e(h)→0`(`h` = cell 尺寸)。
**`Enc` 如何算与定理无关**;§7 给三种 instantiation。

### 1.5 可靠隶属(A4)与有限动作菜单(A5)
`A4`:`box ⊆ ⋃R` 的检查保守(闭 cell、`θ` wrap-around、浮点向外取整)。
`A5`:`U_menu ⊂ U` 有限,运行时可精确执行。

> **信任锚总览:** 安全只依赖 {A3 包络, A4 隶属, A2 种子证书}。`V_θ/Q_θ/f̂` 的"正确性"**不是**信任锚——它们要么只缩小集合(`V_θ` 的 B1 约束、§5),要么只被验证器检查(`Q_θ` 提议)。这正是"新增区域为何 sound"的根本答案。

---

## 2. 扩张算子:从下界 lfp,而非从上界 gfp

子集格 `(2^Grid, ⊆)` 上定义验证版前驱与生长算子:
$$\mathrm{Pre}_{\mathrm{ver}}(R)=\Big\{c:\ \exists u\in U_{\mathrm{menu}},\ \mathrm{Enc}(c,u)\subseteq \textstyle\bigcup R\Big\},\qquad
G(R)=R\cup\big(\mathcal K\cap \mathrm{Pre}_{\mathrm{ver}}(R)\big),$$
其中 `𝒦={c:c⊆K}`。**主结果用 V-锚定变体(B1):**
$$G_V(R)=R\cup\big(\mathcal K\cap \mathcal V_0\cap \mathrm{Pre}_{\mathrm{ver}}(R)\big),\qquad \mathcal V_0=\{c:\mathrm{lb}\,V_\theta(c)\ge 0\},$$
迭代 `R_{k+1}=G_V(R_k)`,`R_0=S_brake`,极限 `R_∞=⋃_k R_k`。
B1 保证 `R_∞ ⊆ {V_θ≥0}∩K`:**认证的是学习对象 `V_θ` 安全水平集的一个 verified 子集**——这一项纯做减法,不影响任何 soundness。

**与失败的 naive racing closure 的精确关系。** 两者满足同一族不动点方程
`S = R_0 ∪ (𝒦∩Pre(S))`,但:
- naive closure 从 `K` 往里削,是 **gfp 逼近**:每个中间迭代是 over-approximation,**只有收敛后才是不变集** → 有限分辨率下把未收敛集当不变集 → `cbv=205`;
- 本算子从 `R_0` 往外长,是 **lfp 逼近**:每个中间迭代是 under-approximation,**本身就已认证**。

同方程、反方向,只有 lfp 方向给出 anytime-sound 迭代。

**wrapping effect 被构造性消灭。** funnel/closure 要把 box 推过多步轨迹(box 膨胀是标准死法);生长迭代**每次只包络一步**,落点立即被已验证 cell 接住。所有 containment 检查都是 one-step。

记 `ℓ(c)=min{k:c∈R_k}`(层数),新增 cell 记见证动作 `u*_c`。

---

## 3. 定理 A:anytime soundness 与分层回退策略

**分层策略 `σ`(量化状态反馈):** 在 `x` 处取 `c=cell(x)`;若 `ℓ(c)=0` 执行 `π♭` 刹停;否则执行 `u*_c`。

> **定理 A (anytime soundness).** 在 A1–A5 下,对任意 `k` 与任意 `x∈⋃R_k`,策略 `σ` 使得**对所有(对抗、全反馈)扰动序列**,闭环轨迹满足
> $$x_t\in K\quad\forall t\ge 0,$$
> 且在至多 `ℓ(cell(x))` 步内进入 `R_0`,至多 `ℓ + H_stop` 步内到达 `v=0` 不动点。
> 特别地,**每一个迭代 `R_k`(无需等收敛)都是 sound 的鲁棒控制不变集**,审计 `cbv≡0` by construction。

**证明(对 `k` 归纳)。**
*基例 `k=0`:* `R_0=S_brake`,由 A2 每点有永久安全的刹停轨迹,结论成立。
*归纳步:* 设结论对 `R_k` 成立。取 `x∈R_{k+1}\R_k`,则 `c=cell(x)∈𝒦∩𝒱_0∩Pre_ver(R_k)`。
1. `x∈c⊆K`(时刻 0 安全)。
2. 执行 `u*_c`。对手任选 `d_0∈D`(可依赖 `x`):由 A3,`f(x,u*_c,d_0)∈Reach(c,u*_c)⊆Enc(c,u*_c)`;由构造与 A4,`Enc(c,u*_c)⊆⋃R_k`。故 `x_1∈⋃R_k` **与 `d_0` 无关**,且 `ℓ(cell(x_1))≤k`。
3. 对 `x_1` 应用归纳假设:其后轨迹永久 `∈K`、`≤k` 步进 `R_0`。
拼接:`x_t∈K ∀t`;层数严格递减保证 `≤ℓ(c)` 步进 `R_0`,再 `≤H_stop` 步入 `v=0`。∎

> **核心观察(为何绕开 Gate 0b)。** 证明中**没有任何**对 `V_θ` 下降、`Q_θ` 正确性的引用。那 15% level-independent 的洞(`min h1≈−0.08`)**根本不参与**这条推理链——不变性来自"已验证动作下的集合包含",不来自学习函数的解析性质。这是从"认证学习 `V` 的 sub-level set 不可能"到"构造性认证可行"的机制本质。

**推论 A.1 (finite-time-to-quiescence).** 每个认证状态有显式上界 `ℓ(x)+H_stop` 的有限时间静止性——比"不变"更强、可单独陈述。

---

## 4. 终止性、最小不动点、序无关、单调旋钮

> **命题 1 (终止 + lfp).** `G_V` 在有限格上**单调**(`R⊆R'⇒G_V(R)⊆G_V(R')`,因 containment 与 `Enc` 对 `R` 单调)且 **inflationary**(`R⊆G_V(R)`)。故 `{R_k}` 单调不减,`≤ N:=|Grid\R_0|` 步稳定到 `R_∞`,且 `R_∞` 是包含 `R_0` 的 `G_V` 的**最小不动点**。

*证.* 单调 + inflationary + 有限格 ⇒ Kleene/Tarski 升链在有限步稳定。任取不动点 `P⊇R_0`,归纳 `R_k⊆P` ⇒ `R_∞⊆P`,故 `R_∞` 最小。∎

> **命题 2 (混沌迭代 / 序无关).** 按任意**公平**顺序逐 cell 异步处理(而非整轮同步扫),收敛到同一 `R_∞`(Cousot chaotic iteration;单调 + inflationary + 有限格)。

工程含义:frontier 队列可乱序、可并行,结果唯一。→ 提供一个干净的可复现实验:Q-ranked 顺序 vs 穷举顺序,**最终集合 bitwise 相同**,只有耗时不同。

> **命题 3 (单调旋钮).** `R_∞` 对种子 `R_0`、菜单 `U_menu`、**网格细化**单调不减,对 `D` 单调不增。

*网格细化部分证明要点.* 设粗 cell `c` 以 `u` 通过(`Enc(c,u)⊆⋃R`)。细化后 `c=⋃c_i`,由 A3 单调性 `Enc(c_i,u)⊆Enc(c,u)⊆⋃R`,而 `⋃R` 的体积被细网格 cell 覆盖,故每个 `c_i` 以同一 `u` 通过 ⇒ **verified 体积不减**。∎
含义:**anytime / 多分辨率**叙事——更多算力 ⇒ 更大认证集,绝不回退。

---

## 5. 饱和表、可部署 Q-CBF shield,与 `Q_θ` 的三个入口

§3 的策略是"层层下楼直到刹停",sound 但过保守(部署叙事不能是"所有车最终都停")。补**饱和后处理**:对收敛后的 `R_∞`,对每个 `c∈R_∞`、每个 `u∈U_menu` 重算
$$A_{\mathrm{ver}}(c)=\{u:\ \mathrm{Enc}(c,u)\subseteq \textstyle\bigcup R_\infty\}.$$

> **引理 5.1 (饱和非空 + 递归不变).** 每个非种子 cell 满足 `u*_c∈A_ver(c)`(containment 对 `R` 单调,`R` 只增),故 `A_ver(c)≠∅`。策略类
> `Π_safe = { 在 c 处任选 u∈A_ver(c);若 A_ver(c)=∅(仅纯种子 cell)则 π♭ }`
> 中**每一个**选择子都使 `R_∞` 不变或进入刹停安全模式。

*证.* 取 `c∈R_∞`,`u∈A_ver(c)`:`∀d, f(x,u,d)∈Enc(c,u)⊆⋃R_∞`,即后继留在 `R_∞`;对后继 cell 递归用 `A_ver`,落不到 `A_ver=∅` 的 cell(那些只在纯种子层且由 `π♭` 兜底)。归纳即得不变。∎

### 部署对象 = certified shield × 学习选择器(主线 claim 的最终形态)
运行时:在 `A_ver(cell(x)) ∩ Φ_θ(x)` 内取 `Q_θ`-argmax;若交集空则取 `A_ver` 内任一(下楼);若 `A_ver` 空则 `π♭`。
- **safety** 由认证表 `A_ver`(= 安全约束)保证;
- **performance/style** 由 `Q_θ` 在已验证动作内决定。
赛车行为是**递归的**(可永远 race,刹车只是兜底层),narrowing 被解掉。

### `Q_θ` 恰好三个入口,全在安全链之外
1. **提议(离线加速)**:逐 frontier cell 把菜单按 `Q_θ(center)` 排序,先试 top-`m`,失败再穷举。提议错了 verifier 拒掉 ⇒ **over-permissive 无害**。
2. **运行时选择(在线)**:已验证动作集内的偏好排序 = "被认证约束着的 `Q`-引导赛车"。
3. **度量(报告)**:
   - `live_frac = |{c: A_ver(c)∩Φ_θ ≠ ∅}| / |R_∞|`;
   - `false-feasible mass`(下)。

> **定理 S (false feasibility / 学习谓词 over-permissiveness — 定量版).** 设
> $$\mathcal F=\Big\{(c,u):\ u\in\Phi_\theta(\mathrm{center}(c)),\ \mathrm{Enc}(c,u)\not\subseteq \textstyle\bigcup R_\infty\Big\}.$$
> 若 `mass(𝓕)>0`,则未限制的 `Φ_θ` 滤波器**不是** `R_∞`-不变:存在 `(x,u),u∈Φ_θ(x)`,其鲁棒后继逸出认证集,在对抗 `d` 下违反约束。即审计必得 `cbv>0`。
> 这把现有 RFC 的"存在反例(N=540, cbv=205)"升级为**对学习 Q-CBF artifact 的定量刻画**,并构成"后验验证必要性"(necessity)的正式论证。

*证.* 取 `(c,u)∈𝓕`。`Enc(c,u)⊄⋃R_∞` ⇒ 存在 `x∈c,d∈D` 使 `f(x,u,d)∉R_∞`(由 A3 的紧性与包络非平凡;实现上由"box 触到未认证 cell"见证)。该后继落在认证集外,后续无 sound 安全证书,greedy-`d` rollout 即把它推向 `g<0`。∎

> **claim 对齐说明.** `A_ver` 锚定在 `R_∞⊆{V_θ≥0}∩K`(B1),且部署谓词是 `Φ_θ`——**认证的就是部署的那个学习 Q-CBF 滤波器**,只是把它**限制到 verifier 认可的子动作集**。grow 迭代是"discharge 这张后验证书"的引擎,不是替换学习对象。

---

## 6. 定理 C:相对完备性 — 极限对象是 robust capture basin

理想(无保守性)迭代 `C_{k+1}=C_k∪(K∩Pre(C_k))`,`C_0=R_0`,极限
$$C_\infty=\mathrm{Capt}_K(R_0):=\{x:\ \exists\,\text{策略},\ \forall d,\ \text{轨迹}\subseteq K\ \text{且有限步进}\ R_0\}$$
(Aubin capture basin 的 discriminating/对抗版)。三层夹逼:
$$R_\infty\ \subseteq\ \mathrm{Capt}_K(R_0)\ \subseteq\ \mathrm{Viab}(K).$$
左 = 定理 A;右 = "在 `K` 内走到 viable 的 `R_0`"本身 viable。

> **定理 C (相对完备性,陈述).** 设 `f` 对 `(x,u)` Lipschitz(常数 `L_f, L_u`),包络 overshoot `e(h)`,动作网密度 `η`。定义 `δ`-鲁棒捕获域 `Capt^δ`:存在策略使所有访问状态 `∈ K⊖δB`,且每步鲁棒后继到下一层目标的间隙 `≥δ`。若
> $$L_f\,h + e(h) + L_u\,\eta\ \le\ \delta,$$
> 则每个完全落在 `Capt^δ` 内的 cell 都被 `R_∞` 验证。于是 `vol(Capt^δ \ R_∞)→0`;若 basin 边界正则(`vol(Capt\Capt^δ)→0` as `δ→0`),则
> $$\mathrm{vol}\big(\mathrm{Capt}_K(R_0)\setminus R_\infty\big)\ \xrightarrow[h,\eta\to 0]{}\ 0.$$

**证明骨架(margin 归纳).** 对 `δ`-捕获指标 `m` 归纳。cell `c` 全在指标 `≤m` 区域:取中心 `x_c` 的 `δ`-鲁棒动作 `u`,菜单近邻 `ũ` 替换,则
$$\mathrm{Enc}(c,\tilde u)\subseteq f(x_c,u,D)\ \oplus\ \big(L_f h+e(h)+L_u\eta\big)B\ \subseteq\ \{\text{指标}\le m{-}1\ \text{区域,}\ \delta'\text{-深}\}.$$
该区域内被 successor box 触到的 cell 全落在 `Capt^{δ'}_{m-1}`,由归纳假设已验证 ⇒ containment 通过 ⇒ `c` 在第 `≤m` 波加入。常数簿记标准但繁琐。∎

> **诚实定位(novelty 边界).** "网格 backward-reachable 收敛到 capture/viability kernel"**不是**本工作的 novelty——与 Saint-Pierre viability kernel 逼近、Cardaliaguet–Quincampoix–Saint-Pierre discriminating kernel、Reissig 等 feedback-refinement-relation 的 relative completeness **同构**。本工作的 delta 是这些性质被**整合到一个 post-hoc、anytime-sound、种子锚定在已认证学习对象 (`V_θ`/`π♭`)、`Q_θ` 引导提议、并配 false-feasibility 负结果**的认证流程里(见 §0)。**论文里必须把 Theorem C 作为"我们的方法继承的已知收敛保证",而非原创定理。**

> **推论 C.1 (种子丰富性 ⇒ 逼近最大不变集).** 若每个 `δ`-内部 viable 状态都能 `δ`-鲁棒地在 `K` 内到达 `R_0`(**BR 假设**),则 `R_∞→Viab(K)`(测度意义)。

车模物理论据:kinematic bicycle 曲率 `κ=tanδ_s/L` 与 `v` 无关 ⇒ 边刹边满舵走同一圆弧(只是更慢),"viable ⇒ 能减速入种子"几乎成立;唯一破绽:低速下单位弧长暴露给扰动的**时间**更长。→ 把假设变成**可测残余 gap 分解**(§9 诊断):
$$\Omega^*\setminus R_\infty\ \approx\ \underbrace{\mathrm{Viab}\setminus \mathrm{Capt}}_{\text{种子不足}}\ \cup\ \underbrace{\text{离散化}}_{h}\ \cup\ \underbrace{\text{包络保守}}_{e(h)}.$$

> **gap 未闭合时的杠杆:丰富种子。** `R_0` 取多 funnel 并(刹车 × 左/右满舵 × 低速 loiter 圈),每个按现有 `S_brake` 流程验证,其余不变(命题 3 保证只增不减)。

---

## 7. 通用性:抽象接口的三级 instantiation(高维 / `f` 不可建模)

§3–§6 **唯一**用到的动力学性质是 A3(`Enc ⊇ Reach`)。故同一套定理对任何能提供 `Enc` 的系统成立。三级 soundness:

| 级 | 适用 | `Enc` 怎么算 | soundness | overshoot `e(h)` |
|---|---|---|---|---|
| **D 解析级** | toy(Dubins/F1TENTH),`f` 解析 | `f` 逐维区间/中心形式(`cos/sin` 在 `θ` 区间取精确极值) | 确定性 sound | `O(h)`,中心形式 `O(h²)` |
| **M 模型级** | 高维,`f̂` 学习但有可靠误差 | CROWN on `f̂` ⊕ 误差集 `ℰ` ⊕ `D` | 确定性 sound(若 `ℰ` 可靠) | `O(h)+‖ℰ‖` |
| **P 统计级** | 高维,`f` 黑箱 | `f̂` + conformal/scenario 残差界 `ℰ_δ` | sound **w.p. ≥ 1−δ** | + `‖ℰ_δ‖` |

> **统一陈述:模型误差 = 额外扰动。** 设可靠界 `f_true(x,u)∈f̂(x,u)⊕ℰ(x,u)`。令 `D_eff` 使得
> $$\{f_{\mathrm{true}}(x,u,d):d\in D\}\subseteq \hat f(x,u)\oplus \mathcal E(x,u)\oplus D\text{-effect}=:\mathrm{Reach}_{\mathrm{eff}}(\cdot),$$
> 则把 `ℰ` 吸收进扰动集即可。**"robust Q-CBF"逐字升级为"对 扰动 + 模型误差 同时鲁棒"**,定理 A/C/S 无需改动,只是 `Enc` 的来源变了。

**P 级的 δ 簿记(必须诚实写清).** 两种做法:
- **逐检查 union bound**:`#{(c,u) checks}` 很大,`δ` 被稀释——不推荐;
- **全局残差界(推荐)**:对残差函数 `r(x,u)=f_true−f̂` 做**一次** conformal/scenario,得 sup-范数界 `‖r‖≤ρ_δ` w.p. `≥1−δ`,**对所有 cell 检查共用同一 `δ`**。这样 §3–§6 全部命题以"w.p. ≥1−δ"成立,且 `δ` 不随网格膨胀。

> **claim 迁移说明(诚实).** D 级认证"true `f`";M/P 级认证的是"`f̂` + 可靠/PAC 误差的鲁棒过近似"。这是 soundness 在高维下的**精确含义**,论文必须明写——它**不削弱** post-hoc / robust / Q-CBF 主线,只是把"true `f`"替换成"sound over-approximation of true `f`'s reachable set",并把模型误差并入鲁棒性预算。

> **通用性 demo 的设计含义(对应提醒 2)。** 后期高维系统选 `f` 不可解析建模者(如接触丰富 manipulation、软体、或高维 quadrotor 带气动),用 M/P 级 instantiate。被认证的 artifact 此时是 `(V_θ, Q_θ, π♭, f̂)` 全冻结网络——**仍是同一句 post-hoc certification of a frozen learned robust Q-CBF artifact**,只是抽象接口从区间算术换成 CROWN(`f̂`)+ conformal(`ℰ`)。

---

## 8. 实现层:逐环节 soundness 落地

**包络(A3).** D 级:解析 `f` 逐维区间求值,浮点向外取整(`np.nextafter`)或所有比较减固定 `ε`。M/P 级:CROWN 批量过 `f̂`,叠 `ℰ`。
**单测(沿用 D1–D4 风格):** 百万级 Monte-Carlo,随机 `x∈c,d∈D`,断言 `f(x,u,d)∈Enc(c,u)`,失败率进 CI gate。

**隶属(A4).** successor box → 整数索引范围(向外扩 1 ulp),`θ` 取模;被触到的 cell 必须全 `∈R`;任一维出网格域(除 `θ` wrap)即该 `(c,u)` 失败。`v` 维注意 `a_max` 顶出域 → 菜单 state-dependent 裁剪(仅 `v+aT≤v_max` 处允许 `a_max`)。

**主算法:counter-based 混沌迭代(Dijkstra 式,零重复验证).**
`(c,u)` 失败时**不丢弃**,记其 **blocking cells**(successor box 触到的、尚不在 `R` 的 cell),建反向索引 `b ↦ 等它的 (c,u) 列表`。`b` 入 `R` ⇒ 递减等待计数;某 `(c,u)` blocking 清空 ⇒ 通过 ⇒ `c` 入 `R`(若 `∈𝒦∩𝒱_0`),记 `u*_c, ℓ(c)`,唤醒以 `c` 为 blocker 者。每个 `(c,u)` 只算一次包络。

```text
R ← S_brake;  每个候选 cell 的菜单按 Q_θ(center) 排序
for c ∈ K_grid \ R, u ∈ menu(c):        # 可只先试 Q-top-m
    B ← Enc(c,u)                        # D级:解析区间;M/P级:CROWN(f̂)⊕ℰ⊕D,逐 action 向量化
    if B ⊄ Domain: discard (c,u); continue
    blockers ← cells(B) \ R
    if blockers = ∅: promote(c,u)
    else: register (c,u) waiting on blockers
promote(c,u): R += c; record u*_c, ℓ(c); wake waiters-on-c   # B1: 先过 lb V_θ(c) ≥ 0
loop until queue empty → R_∞
saturate: 对 R_∞ 重算 A_ver(c);batched CROWN 出 live_frac / false-feasible mass;导出部署表
```

**算力.** 候选 cell `~1e4–1e5`(res44/56),菜单 15–27 动作,D 级是 numpy 区间算术(每 action 一次全场向量化)→ **分钟级,远轻于 CROWN funnel**;CROWN 只剩 B1 的 `lb V_θ`、M/P 级的 `f̂`、与后处理度量。

**证书工件(formal-methods 加分项).** 产物 = 表 `{(c, u*_c, ℓ(c))} ∪ 种子证书`,可由 ~50 行独立 checker 重验(逐条重算包络 + 隶属 + 层序)。"证书小且可独立机检"是审稿吃的点;`audit`(shield 策略下 extremal + greedy-`d` rollout)保留,**`cbv` 恒 0,任何非零 = 包络/索引 bug**,直接做 CI gate(把 audit 从"测试"升格为"对 checker 的交叉验证")。

---

## 9. 实验技术框架

### P0 — sanity(2–3 天)
包络 MC 单测;wrap/索引边界测试;小网格 end-to-end:验证集内对抗 `d` 暴力模拟(零违规),并与小实例**精确离散 capture basin**(暴力 fixpoint)对照,标定保守性。

### P1 — go/no-go(约 1 周,res 44)
**唯一问题:`ρ` 涨不涨、涨多少。**
输出:`ρ(k)` 曲线、`ρ_∞` vs `ρ_brake=0.80`、`ℓ` 直方图、Q-top-`m` 首试通过率 vs 穷举(量化 `Q_θ` 提议价值)、耗时;frontier 失败原因分解(出 `K` / 出域 / 被未认证 cell 阻塞 / `𝒱_0` 失败)→ 区分"种子不足"与"保守性"。
**判据:** `Δρ≥+0.05` 继续;`≥+0.10` 强信号;`≈0` 先跑诊断 + 多 funnel 种子再判,不直接放弃。

### P2 — 全量结果(2–3 周)
- res 56(可加 80)复刻 resolution-stability;headline 表加一行:`R_∞`(B1)verified yes / `ρ_∞` / `cbv 0`,与 `0.635`(apparent, cbv 205)、c2fix(cbv 3144)同表。
- 饱和表 + 部署 rollout 三方对比(**certified shield vs naive Φ_θ vs 纯刹车**):`cbv`、racing 时间占比、progress 指标。
- `R_∞` 上 `live_frac` 与 `false-feasible mass`(Theorem S 定量版)。
- **Ablations:** B1 vs A(学习 `V` 是否瓶颈)/ Q-ranked vs 穷举(最终集 bitwise 相同、耗时不同 = 命题 2 实证)/ 种子缩放 / 菜单丰富度扫描。
- **图:** 洋葱图(cell 按 `ℓ` 着色,`v`-slices)/ 双分辨率 `ρ(k)` / `Ω*`–`S_brake`–`R_∞` 三层切片 / audit bar 扩为 `0 / 205 / 3144 / 0`。

### P3 — 通用性 demo(stretch,决定档次,对应提醒 2)
**`f` 不可解析建模的高维系统**(M/P 级 instantiation):学习 `f̂` + conformal `ℰ_δ`,认证 frozen `(V_θ,Q_θ,π♭,f̂)`。报告:`ρ_∞`、`cbv`(w.p. ≥1−δ)、`ρ_∞(h)` 随网格的趋势作为定理 C 实证;与 toy 共用**同一套定理与算法**,只换 `Enc` 后端 → 直接证明系统 + 理论的通用性。

### 风险与对策(按概率)
1. **涨幅小** → 看失败分解;种子不足上多 funnel,保守性局部 split `θ/v` 或中心形式包络。
2. **cell 级常值动作在刀锋区不够** → 上 2–3 步 verified macro-action(motion primitive,包络仍只复合 2–3 步)。
3. **`Ω*` 非真 viability kernel** → 小实例 HJ/暴力 kernel 校准分母;同时报 `|R_∞|/|Ω*|` 与绝对体积;检查 `R_∞\Ω*` 应为空(非空 = 重大发现或 bug)。
4. **P 级 δ 膨胀** → 用全局残差 conformal,勿逐检查 union bound(§7)。

---

## 10. 与现有 RFC 的衔接 / "claim 不漂移"清单

- `S_Q^∞`(RFC 中留位:"future certified racing fixed point")= **本文 `R_∞` 的饱和版**,直接接管该名字,叙事无缝。
- 主线一句话不变:**post-hoc certification of a deployed learned robust Q-CBF artifact**;本模块只把"certificate 怎么 discharge 到接近最大安全体积"补全。

**✅ 本模块声明:**
- `R_∞`(`V_θ`-锚定、`π♭`-种子、grow-from-seed)是 anytime-sound 的鲁棒控制不变集,`cbv≡0`,resolution/分辨率单调。
- 部署谓词仍是学习 `Q_θ`(`Φ_θ`),被限制到 verifier 认可子动作 → narrowing 解除,递归 racing。
- 未限制 `Φ_θ` 被同一 verifier 拒绝(Theorem S,定量 false feasibility)。
- 抽象接口 f-agnostic,高维 / `f` 不可建模经 M/P 级 instantiate,定理逐字成立(soundness 分级)。

**❌ 本模块不声明:**
- 不声明认证了学习 `V_θ` 的 sub-level set(那条死于 Gate 0b 的 level-independent holes)。
- 不把"grow/refine 收敛到 capture basin"当原创(Theorem C 是**继承**的已知收敛类,见 §6 定位)。
- 不把训练损失当证明假设——只有 post-hoc verifier 证明安全。
- M/P 级"sound"是"对 `f̂`+可靠/PAC 误差的鲁棒过近似 sound",非"对真 `f` 解析 sound"——须明写。
