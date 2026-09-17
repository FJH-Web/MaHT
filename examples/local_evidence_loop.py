"""Offline example using a fixed parser; production uses DeepSeek at the boundary."""

from statagent.agents.parser_agent.deepseek import HypothesisParserBackend
from statagent.app.pipeline import ExecutionInput, build_default_app
from statagent.protocol.statistical_object import HypothesisState


class ExampleParser(HypothesisParserBackend):
    def parse(self, user_text: str) -> list[HypothesisState]:
        return [
            HypothesisState(
                id="H001",
                description="Treatment and control means differ",
                metadata={
                    "category": "Parametric",
                    "retrieval_terms": ["independent t-test", "two independent groups"],
                },
            )
        ]


app = build_default_app()
app.parser.backend = ExampleParser()
report = app.run(
    "Compare treatment and control",
    lambda message: ExecutionInput(
        data={"group_a": [4.2, 5.1, 5.4, 6.0], "group_b": [2.1, 2.7, 3.2, 3.5]}
    ),
)
print(report)
