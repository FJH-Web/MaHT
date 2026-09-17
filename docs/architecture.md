# StatAgent v0.1 Architecture

StatAgent is organized around evidence-bearing hypotheses. DeepSeek has four bounded semantic roles: converting the user's request to validated `HypothesisState` instances; selecting one method from HSR-retrieved Method Card candidates and proposing a validated data mapping; interpreting immutable numerical results and locally generated figures; and composing structured paper sections. It is explicitly forbidden from producing executable analysis code or modifying numerical results.

After parsing, every agent receives and returns a `StatisticalObject`. The Hypothesis Manager maintains a DAG, HSR retrieves up to five Method Card leaves, and the semantic matcher can choose only from that candidate set. A validated data adapter binds dataframe columns to the selected card's input roles. The dispatcher verifies that the card is registered before invoking its pre-deployed local function. Method Card visualization recipes generate local PNG artifacts and semantic manifests. DeepSeek interprets those immutable artifacts, and the latest state is persisted to Statistical Memory. AHS schedules newly unblocked hypotheses by expected information gain minus cost.

The report pipeline asks DeepSeek for named manuscript sections—not arbitrary HTML. A deterministic, escaped `report/templates/paper.html` template embeds the locally generated figures as base64, so the downloadable report is self-contained and its layout is reproducible.

The loop is finite: every parsed hypothesis is executed at most once per run. Browser agents, autonomous tool agents, MCP, generated statistical code, multi-LLM negotiation, and unbounded loops are deliberately out of scope.

## Extension boundaries

- Replace `EmbeddingProvider` and `VectorStore` to use a local embedding model and a production vector database.
- Replace `DecisionPolicy` to implement Bayesian or domain-specific posterior updating.
- Replace `Scheduler` with MARL while preserving its public interface.
- Replace JSON persistence behind `StatisticalMemory` with a database adapter.
- Add methods by constructing a complete Method Card and registering one pre-deployed function; no router edits are needed.
