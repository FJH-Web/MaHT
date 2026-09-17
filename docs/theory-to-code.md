# Theory-to-Code Mapping

| Paper construct | Mathematical object | Engineering implementation | Primary invariant |
|---|---|---|---|
| Hypothesis State | `H_i = (d_i, s_i, c_i, p_i, E_i, q_i, D_i)` | `HypothesisState` | Confidence and posterior remain in `[0, 1]` |
| Hypothesis Graph | `G_H = (V_H, E_D)` | `HypothesisGraph` | Dependency edges form a DAG |
| Statistical Decision Protocol | `SDP = (H, E, M, R, P, D, C, T)` | `StatisticalObject` | The sole inter-agent message type |
| Hierarchical Statistical Retrieval | `HSR(H_i) -> M_k` | `HierarchicalRetriever`, `MethodCard`, `StatisticalMethodRegistry` | Retrieves metadata, never code |
| Statistical Memory | `H -> M -> R -> D` lineage | `StatisticalMemory` | Append-only revision history, JSON persistence |
| Active Hypothesis Scheduling | `U(H_i) = EIG(H_i) - Cost(H_i)` | `Scheduler`, `ActiveHypothesisScheduler` | Highest current utility is scheduled first |
| Local statistical execution | `A_k(X, theta) -> R_k` | `StatisticalDispatcher` and registered functions | Only pre-registered APIs can execute |
| Semantic method decision | `argmax_{M_k in C(H_i)} P(M_k | H_i, X_schema)` | `SemanticMethodMatcher` | LLM may choose only from HSR candidate set |
| Statistical visualization | `V_{kj} = g_{kj}(X, R_k)` | `VisualizationEngine` and Method Card recipes | Figures are local functions of data and immutable results |
| Result interpretation | `I_i = LLM(H_i, M_k, R_k, V_k)` | `InterpretationAgent` | LLM cannot modify `R_k` or execute code |
| Paper composition | `P = T(LLM(SM), V)` | `PaperReportComposer` + `paper.html` | LLM writes section prose; local template owns HTML |

The docstrings on each core class repeat its paper construct so that manuscript terminology, implementation names, and experimental logs remain aligned.
