# Peer Review (R2): "Minimal Diagnostic Principles of Life: Criterion-Ablation and Observable Surrogate Selection in a Digital Ecosystem"

**Venue:** The 2026 Conference on Artificial Life (ALife 2026)  
**Submission Type:** Full Paper  
**Review Round:** Revision (R2)  
**Recommendation:** Accept  
**Reviewer Confidence:** 4/5  
**Overall Score:** 9/10

---

## 1. Summary

本論文は、生物学の教科書が掲げる7つの生命基準（細胞組織、代謝、恒常性、成長・発達、生殖、刺激応答、進化）を機能的に相互依存する計算プロセスとして単一のALifeシステムに統合し、各基準の機能的必要性をcriterion-ablation実験により検証する。改訂版では、以下の大幅な拡張が加えられている。

Phase 2 powered surrogate analysis（200条件、5環境レジーム、14,000シミュレーション）が追加され、validation R²=0.955、out-of-regime test R²=0.950を達成。関連研究が8システムに拡充（Polyworld, Geb, Stringmol, Beer '04を追加）され、ablation-testable列を含む比較テーブルが提示された。進化基準に対してcrossover演算子、neutral-drift control、fitness landscape分析（方向性選択H1の棄却を正直に報告）が追加された。さらに、死因分析（§S8）、閾値感度分析（§S10）、境界トポロジー比較（§S10）、NNコントローラー感度分析（§S12）、エンジン間比較（§S13）、成長/生殖分離実験（§S14）が新たなSupplementary Materialとして追加された。

全体として、前回の査読で指摘した主要な懸念点のほぼすべてに対して実質的な改善が施されており、論文の完成度は飛躍的に向上している。

---

## 2. 前回指摘事項への対応評価

### W1（進化基準の成熟度不足）→ 大幅に改善

改訂版は進化基準に対して多角的な追加分析を行っている。

Crossover実験（segment-wise / uniform）とneutral-drift control（親選択をランダム化し、mutationは維持）の追加は、genuine adaptationとneutral driftの識別という前回の中核的懸念に直接対応している。Figure 16はcrossover条件とno-crossover条件の個体群ダイナミクスを明確に可視化しており、neutral driftが進化ありの条件とは統計的に異なるエネルギー軌跡を示すことが確認できる。

Fitness landscape分析（§S9）では、pre-specified hypothesisであるH1（方向性エネルギー選択）がp=0.944で棄却されたことを正直に報告しつつ、H2（遺伝率）でparent-offspring回帰勾配b̂=0.747（95% CI [0.730, 0.763]、n=68,497ペア）という強い証拠を示している。「選択がエネルギー以外の形質（空間効率、生殖タイミング）に作用しているか、エネルギーが安定化選択の下にある」という解釈は科学的に誠実であり、過大主張を避けた優れた議論である。

進化をLevel 3に留めると明言し、Level 4達成にはopen-ended dynamicsの実証が必要であると認めている点も、前回の懸念に対する適切な対応である。

**残存する懸念:** エネルギー以外の形質（空間効率、生殖タイミング）に対する選択の証拠は依然として間接的であり、直接的な測定はない。neutral-drift controlとevolution条件の個体群レベル差は示されているが、どの具体的形質が選択の標的であるかの特定は今後の課題として残る。ただし、これは現論文のスコープ内で合理的に対処可能な範囲を超えており、future workとして適切に位置づけられている。

### W2（細胞組織のスカラー簡約化）→ 部分的に改善

§S4にスカラーbとswarm agentsの空間ダイナミクスの関係についての説明が追加された。「bはswarm agent損失の機能的帰結を追跡する」「spatial_cohesion_meanは独立な空間メトリクスとしてbの変化が真の空間的解離に対応することを確認する」という記述は、前回の懸念に対する概念的な回答を提供している。

「スカラー表現はクリーンなアブレーション実験を可能にするための意図的な簡略化」というフレーミングは合理的であり、空間cohesionメトリクスによる検証がこの簡略化の妥当性を支持している。

**残存する懸念:** bがswarm agentsの空間配置から計算されるのか、それとも独立に更新されるのかという具体的なメカニズムの記述は依然として完全ではない。ただし、これは実装の詳細であり、論文の主張を損なうものではない。

### W3（基準縮約分析のパワー不足）→ 完全に解決

Phase 2 powered analysis（200条件、5環境レジーム、14,000ラン）の追加は、前回の最大の統計的懸念を完全に解消している。LASSO/Elastic Net stability selectionで11/12の候補特徴量がstability frequency ≥ 0.98で保持され、train R²=0.954、validation R²=0.955（見たレジーム、未見シード）、out-of-regime test R²=0.950（未見レジーム）を達成している。permutation検定（p<0.001、帰無95パーセンタイルR²=0.004）は統計的有意性を確認し、alive-AUC-onlyベースライン（R²=0.921）との比較はmulti-surrogate benefitを定量化している。

3層のregime-based split（train/validate/test）と環境レジームを一般化軸として使用する設計は、単なるサンプルサイズ拡大を超えた方法論的進化である。

### W4（関連研究の網羅性）→ 大幅に改善

Table 1が8システムに拡充され、Polyworld (Yaeger, 1994)、Geb (Channon, 2001)、Stringmol (Hickinbotham et al., 2011)、Beer '04 (Beer, 2004) が追加された。「Abl.」列（ablation-testable）の追加は、本論文の方法論的貢献を視覚的に強調する効果的な工夫であり、前回の改善提案を超えた好判断である。自己評価バイアスへの明示的acknowledgment（「Independent cross-validation by system developers would strengthen these comparisons」）も適切。

5段階ルブリックの定義がTable 6（§S4）に明示された点は、前回の懸念を完全に解消している。

### W5（スケーラビリティ）→ 部分的に対応

Limitationsセクションに「Mac Mini M2 Pro; 10× scaling is feasible with spatial partitioning」との具体的記述が追加された。大規模シミュレーションの実施はされていないが、計算資源の制約が明示された点は透明性の向上として評価できる。

### W6（成長/生殖分離可能性）→ 完全に解決

§S14の4条件実験（normal, no growth, bypass maturity gate, no growth + bypass）は、前回の改善提案（C1最低限に含まれていた「maturity gate bypass」条件）を正確に実施している。bypass vs. no-growth-bypass: Δ=+148.2, U=900, p<0.001という結果は、growthがreproduction gatekを超えた独立効果（boundary repair efficiency、sensing range、metabolic throughput）を持つことを定量的に実証しており、極めて説得力がある。

---

## 3. 新たに追加された分析の評価

### 死因分析（§S8）— 高く評価

Figure 17の死因分布（boundary collapse / energy depletion / age limit）は各アブレーション条件の失敗メカニズムを明確に可視化している。特にFigure 19の「essential triad」に対するfailure cascade分析——代謝アブレーションではエネルギーが最初に崩壊（median break step 20）し、生殖アブレーションではエネルギーと境界が安定のまま人口統計的消耗で失敗する——は、coupling graphの解釈に直接的なmechanistic insightを提供している。これは前回の改善ロードマップでC3（即実行可能、コスト低）として推奨した項目であり、期待通りの効果を発揮している。

### 閾値感度分析（§S10）— 高く評価

12の閾値組み合わせ中9つでSpearman ρ=1.0、残り2つでρ=0.5（代謝と応答のnear-tieによるrank swap）という結果は、アブレーション効果階層がパラメータ選択に対してrobustであることを示す強い証拠。Figure 21のヒートマップは一目で結果を把握でき、可視化としても優れている。

### 境界トポロジー比較（§S10）— 高く評価

Toroidal vs. bounded環境でΔ%ランク順位が完全に保存される（ρ=1.0）という結果は、前回のQ4（toroidal境界条件の影響）に対する決定的な回答。bounded環境が全体的により厳しい（organisms cannot wrap around walls）が相対的階層は不変という知見は、フレームワークの一般性を支持する。

### NNコントローラー感度（§S12）— 適切

Hidden層サイズ{8, 16, 32}でKruskal-Wallis p=0.100（有意差なし）という結果は、NN容量がボトルネックでないことを示す。ただし、この実験はablation効果階層そのものの感度（NNサイズを変えた上でablationを行い、Δ%のランクが保存されるか）ではなく、baseline性能のNN依存性のみを検証している点に注意。この区別はminor issueだが、記述の精密化が望ましい。

### エンジン間比較（§S13）— 高く評価

Counter/Toy/Graphの3エンジン全てで7基準アブレーションが有意（6/7基準でδ=1.00）、進化が全エンジンで最弱（Counter: δ=0.55, Toy: δ=0.61）という結果は、基準の必要性がシステムレベルの性質であり代謝実装のアーティファクトでないことを強く支持する。

### Fitness landscape分析（§S9）— 極めて高く評価

H1棄却（p=0.944）を正直に報告した上で、H2成功（b̂=0.747）との組み合わせから「選択がエネルギー以外の形質に作用しているか、安定化選択」という解釈を導出する流れは、科学的誠実さの模範である。pre-specifiedな仮説のnegative resultを透明に提示する姿勢は、ALife分野における再現性危機への健全な対応として高く評価する。

---

## 4. Strengths（改訂版における新たな強み）

### S1: Phase 2分析の方法論的成熟度

200条件 × 5レジーム × 70シードによる14,000ランは、Phase 1の24条件600ランから23倍以上のスケールアップであり、regime-based splitによるout-of-regime一般化テスト（R²=0.950）は「observable surrogatesが未知のパラメータ領域でも生命らしさを予測できる」という主張に対して十分な証拠を提供する。permutation null（95パーセンタイルR²=0.004）との差は圧倒的であり、統計的有意性に疑いの余地がない。

### S2: Ablation side effectsの明示的議論

「knockout semantics」のフレーミング——アブレーションはdownstream cascadeを含むシステムレベルの帰結を測定する、分子生物学のgene knockoutにおけるpleiotropic effectsと同様——は、潜在的な批判に対する先手を打った巧みな議論である。off-target effect table（Table 5）との組み合わせにより、各アブレーションの直接効果と間接効果が定量的に分離されている。

### S3: アブレーションのエネルギー再分配非発生の明示

「skipping a process does not redistribute saved energy or computation to other processes」という明記は、アブレーション実験の解釈上の重要な前提を明確化している。この一文がなければ、「プロセスを停止すると余ったリソースが他のプロセスを強化する」という代替説明が残り得た。

### S4: Functional analogy across criterion typesの議論

連続的基準（代謝、恒常性）とイベント駆動型基準（生殖、進化）に対する「動的プロセス」条件の適用の違いを議論し、「standing readiness」（エネルギーリザーブや遺伝的変異）としてのsustained functional engagementという解釈を提示した点は、フレームワークの概念的精密化として有意義。

### S5: Supplementary Materialの網羅性

§S1–S14の14セクション（mid-run, ecology stressor, phenotype clustering, system design, pairwise ablation, evolution timescale, full statistics, failure mode, fitness landscape, sensitivity/robustness, graded/cyclic, NN sensitivity, engine comparison, growth/reproduction separability）は、考え得るほぼすべての批判に対して先行的に証拠を提供している。このレベルの補足資料は査読プロセスへの深い理解と誠実な取り組みを反映している。

---

## 5. 残存する懸念点

### R1: 進化基準のLevel 3からの引き上げ不達成（Minor）

著者は進化がLevel 3に留まることを明確に認めており、crossoverとneutral-drift controlの追加にもかかわらず、open-ended dynamicsの実証にはis 10⁵+ステップが必要と正しく指摘している。Level 3は「Dynamic process with measurable degradation upon removal」であり、現在の結果（δ=0.39 at 2,000 steps、d=1.42 at 10,000 steps）はこの定義を満たしている。しかし、7基準中6基準がLevel 4で1基準がLevel 3という非対称性は、「7基準すべての統合」という主張にやや影を落とす。

著者はこの限界を十分に認識しており、future workの方向4に明記している。改訂版ではこれ以上の対応は不要だが、進化のLevel 4達成は後続研究の最優先課題として位置づけるべきである。

### R2: Phase 2のseed重複（Minor）

Phase 2のデータ分割において、seeds 40–69がvalidation（regimes A–C）とtest（regimes D–E）の両方に出現する。著者は「to maximise statistical power within each regime」と説明しているが、これによりvalidationとtestの独立性がやや損なわれる。一般化軸が環境レジームであるため、seed重複の影響は限定的と考えられるが、完全な独立分割での結果も参考値として報告されるとより説得力が増す。

### R3: alive-AUC-onlyベースラインとの差の解釈（Minor）

Multi-surrogate model（R²=0.955）とalive-AUC-only baseline（R²=0.921）の差は3.4ポイントであり、著者自身が「population viability being the dominant axis of life-likeness」と認めている。この結果は、11特徴量のうち大部分の予測力がalive-AUC一つに集約されることを示唆しており、「minimal diagnostic principle」の実用的含意（少数の代理指標で生命らしさを予測する）に対してやや緊張関係にある。alive-AUCが他の10特徴量の予測力をどの程度包含しているかの分解分析があるとよい。

### R4: 細胞組織のスカラーbとswarm agentsの具体的対応（Very Minor）

§S4の説明は改善されているが、bの更新式（∆b_decay, ∆b_repair）がswarm agentsの空間配置とは独立に計算される点が依然として完全には明確でない。bが空間状態の「summary statistic」なのか「独立変数」なのかを一文で明示すると、読者の混乱が完全に解消される。

---

## 6. Questions for Authors

1. **Phase 2のseed独立分割:** Seeds 40–69がvalidation/testで共有されているが、完全にdisjointな分割（例：validate seeds 40–54, test seeds 55–69）ではR²にどの程度の変化が見られるか？

2. **安定化選択の直接的証拠:** H1棄却とH2成功の組み合わせから安定化選択の可能性が示唆されているが、energy分散の世代間変化（安定化選択下では分散が減少するはず）は測定されているか？

3. **11特徴量の共線性:** Phase 2で11/12特徴量がstability ≥ 0.98で保持されたが、VIFスクリーニングはPhase 2でも実施されたか？11特徴量間の共線性構造はPhase 1の6特徴量構造とどう異なるか？

4. **Out-of-regime一般化のメカニズム:** Regime D–Eの環境パラメータはA–Cとどの程度異なるか？極端に異なるレジーム（例：リソース再生率10倍）でも一般化は保持されるか？

---

## 7. Minor Issues

以下は受理を妨げるものではないが、最終版での改善を推奨する。

**Figure 13のintervention effects表記の不整合:** Figure 13のアノテーションに「metabolism: energy +38%, waste +100%, boundary -41%, internal_state +41%」とあるが、Table 5では「Metabolism: Energy −38%, Waste −100%, Boundary +41%, Int. State −41%」と符号が逆転している。Figure 13のキャプション内での符号定義を確認されたい（Table 5が「ablation時の変化」であれば、代謝ablationでエネルギーは減少するはずであり、Table 5の方が正しい可能性が高い）。

**§S12のNN感度分析のスコープ:** 現在の実験はbaseline性能のNN依存性を検証しているが、ablation効果階層のNN依存性（NNサイズを変えた上で各ablation条件を実施し、Δ%ランクが保存されるか）を検証するとより完全になる。記述にこの区別を明示するか、追加実験を検討されたい。

**Abstract語数:** Phase 2の追加によりAbstractがさらに情報密度を増している。「stability selection」「VIF screening」などの技術的詳細をAbstractから削り、Phase 2の主要結果（R²値と一般化の意味）に焦点を絞ると、初読時の理解が容易になる。

**Conclusion内のフレームワーク適用要件:** 「Minimum requirements for applying the framework are: (1) per-criterion enable/disable toggles; (2) at least one measurable population-level or individual-level outcome variable; and (3) stochastic replication (multiple seeds) for statistical comparison」の追加は非常に有用。これをIntroductionにも含めると、フレームワークの汎用性が冒頭から伝わる。

---

## 8. Overall Assessment

本改訂版は、前回査読で指摘した6つの主要懸念（W1–W6）のうち4つを完全に解決（W3, W4, W5, W6）、2つを大幅に改善（W1, W2）しており、改訂の質は極めて高い。特筆すべき点として、前回の改善ロードマップで提案した項目の多くが実施されている：死因分析（C3）、NNコントローラー感度（C4）、成長/生殖分離実験（W6/S14）、ルブリック定義（F2）、関連研究追加（F3/F1）、比較テーブル拡充（F1）。さらに、提案を超えた独自の改善——境界トポロジー比較、閾値感度分析、crossover/neutral-drift実験、fitness landscape分析のH1棄却の正直な報告——が加えられており、著者の真摯な取り組みが伺える。

Phase 2分析（200条件、14,000ラン、R²=0.950 out-of-regime）はStage 2の統計的基盤を確立し、論文の2本柱（基準統合+代理指標選定）がいずれも十分な深さで展開されるようになった。§S1–S14の14セクションにわたるSupplementary Materialは、考え得るほぼすべての批判に先行的に証拠を提供しており、再現性と透明性の観点で分野の模範となる水準に達している。

残存する懸念は、進化基準のLevel 3（R1）、Phase 2のseed重複（R2）、alive-AUCベースラインとの差の解釈（R3）、細胞組織の実装詳細（R4）の4点であるが、いずれもminorであり、論文の中核的貢献を損なうものではない。

**推奨: Accept.** ALife 2026のfull paperとして、方法論的貢献（再利用可能なcriterion-ablation protocol）、実験的厳密さ（統計設計、robustness検証の網羅性）、科学的誠実さ（negative resultの透明な報告、弱ALifeスタンスの一貫性）のすべてにおいて、会議に求められる水準を十分に超えている。

---

## Score Breakdown

| Criterion | Previous | Revised | Comment |
|---|---|---|---|
| Novelty | 8 | 9 | Phase 2のregime-based一般化設計とcrossover/neutral-drift controlが新規性を追加 |
| Significance | 7 | 9 | R²=0.950 out-of-regime一般化は「minimal diagnostic principle」の実証として説得力がある |
| Technical Quality | 7 | 9 | 14セクションのrobustness検証、Phase 2の14,000ラン、fitness landscape分析のH1棄却 |
| Clarity | 7 | 8 | knockout semantics、criterion type分類、フレームワーク適用要件の追加。Abstractはやや過密 |
| Reproducibility | 9 | 10 | Phase 2のpre-registered decision thresholds、Zenodoアーカイブ、14,000ランの網羅性 |
| Related Work | 6 | 9 | 8システム比較、Abl.列、ルブリック定義、自己評価バイアスの明示的acknowledgment |

**総合: 9/10**（前回7/10から+2）

---

## 10点到達に向けた残り1ポイントのギャップ

スコアを満点にするために残された課題は、本論文の改訂サイクル内というよりも、後続研究で取り組むべき方向性である。

第一に、進化基準をLevel 4に引き上げることが最大の残課題である。10⁵ステップ以上の長期シミュレーションで、Bedau et al. (2000) のActivity StatisticsやTaylor et al. (2016) のOEE指標を計算し、少なくとも「bounded novelty generation」レベルの進化的ダイナミクスを実証できれば、7基準すべてがLevel 4以上となる。

第二に、外部システムへのcriterion-ablation protocolの適用実証がある。LeniaやALIENのような既存オープンソースシステムに同一プロトコルを適用し、どの基準がfunctional analogyを満たし、どれが満たさないかを実験的に示すことができれば、本論文は単一システムの提案を超え、分野横断的な評価フレームワークの実証となる。ConclusionにMinimum requirementsが明記されたことで、この方向性への道筋は既に整っている。

第三に、3,000–10,000個体規模への拡張で創発的生態現象（ニッチ分化、空間的種分化の兆候）が観察されれば、「7基準の統合が個体生存を超えた生態学的複雑性を生む」という暗黙の期待に対する証拠が得られる。

これらは将来の仕事として適切であり、現論文の受理判断を妨げるものではない。

---

*Reviewed: March 2026 (R2)*
