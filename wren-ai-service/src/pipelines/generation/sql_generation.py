import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe
from pydantic import BaseModel

from src.core.engine import Engine
from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.pipelines.common import (
    TEXT_TO_SQL_RULES,
    SQLGenPostProcessor,
    construct_instructions,
    sql_generation_system_prompt,
)
from src.utils import async_timer, timer
from src.web.v1.services.ask import AskConfigurations

logger = logging.getLogger("wren-ai-service")


sql_generation_user_prompt_template = """
### TASK ###
Given user's question, please answer one SQL statement that best answers user's question.

### DATABASE SCHEMA ###
{% for document in documents %}
    {{ document }}
{% endfor %}

{% if exclude %}
### EXCLUDED STATEMETS ###
Ensure that the following excluded statements are not used in the generated queries to maintain variety and avoid repetition.
{% for doc in exclude %}
    {{ doc.statement }}
{% endfor %}
{% endif %}

{{ alert }}
{% if instructions %}
{{ instructions }}
{% endif %}

### FINAL ANSWER FORMAT ###
The final answer must be the JSON format like following:

{
    "sql": <SQL_QUERY_STRING>
}

{% if samples %}
### SAMPLES ###
{% for sample in samples %}
Question:
{{sample.question}}
SQL:
{{sample.sql}}
{% endfor %}
{% endif %}

{% if previous_queries %}
### Previous Queries ###
{% for query in previous_queries %}
SQL Query:
{{query.sql}}
SQL Summary:
{{query.summary}}
{% endfor %}
{% endif %}

### QUESTION ###
User's Question: {{ query }}
Current Time: {{ current_time }}

If previous queries are given, you need to give one different SQL statement that can also answer user's question in different angles.
Let's think step by step.
"""


## Start of Pipeline
@timer
@observe(capture_input=False)
def prompt(
    query: str,
    documents: List[str],
    exclude: List[Dict],
    alert: str,
    prompt_builder: PromptBuilder,
    configurations: AskConfigurations | None = None,
    samples: List[Dict] | None = None,
    previous_queries: List[Dict] | None = None,
) -> dict:
    logger.debug(f"query: {query}")
    logger.debug(f"documents: {documents}")
    logger.debug(
        f"exclude: {orjson.dumps(exclude, option=orjson.OPT_INDENT_2).decode()}"
    )
    logger.debug(f"configurations: {configurations}")
    if samples:
        logger.debug(f"samples: {samples}")
    return prompt_builder.run(
        query=query,
        documents=documents,
        exclude=exclude,
        alert=alert,
        instructions=construct_instructions(configurations),
        samples=samples,
        current_time=datetime.now(),
        previous_queries=previous_queries,
    )


@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql(prompt: dict, generator: Any) -> dict:
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=prompt.get("prompt"))


@async_timer
@observe(capture_input=False)
async def post_process(
    generate_sql: dict,
    post_processor: SQLGenPostProcessor,
    project_id: str | None = None,
) -> dict:
    logger.debug(
        f"generate_sql: {orjson.dumps(generate_sql, option=orjson.OPT_INDENT_2).decode()}"
    )
    return await post_processor.run(generate_sql.get("replies"), project_id=project_id)


## End of Pipeline
class GenerationResult(BaseModel):
    sql: str


SQL_GENERATION_MODEL_KWARGS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "sql_results",
            "schema": GenerationResult.model_json_schema(),
        },
    }
}


class SQLGeneration(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        engine: Engine,
        **kwargs,
    ):
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_generation_system_prompt,
                generation_kwargs=SQL_GENERATION_MODEL_KWARGS,
            ),
            "prompt_builder": PromptBuilder(
                template=sql_generation_user_prompt_template
            ),
            "post_processor": SQLGenPostProcessor(engine=engine),
        }

        self._configs = {
            "alert": TEXT_TO_SQL_RULES,
        }

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(
        self,
        query: str,
        contexts: List[str],
        exclude: List[Dict],
        samples: List[Dict] | None = None,
        project_id: str | None = None,
        previous_queries: List[Dict] | None = None,
        configurations: AskConfigurations | None = None,
    ) -> None:
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/sql_generation.dot",
            inputs={
                "query": query,
                "documents": contexts,
                "exclude": exclude,
                "samples": samples,
                "project_id": project_id,
                "configurations": configurations,
                "previous_queries": previous_queries,
                **self._components,
                **self._configs,
            },
            show_legend=True,
            orient="LR",
        )

    @async_timer
    @observe(name="SQL Generation")
    async def run(
        self,
        query: str,
        contexts: List[str],
        exclude: List[Dict],
        samples: List[Dict] | None = None,
        project_id: str | None = None,
        previous_queries: List[Dict] | None = None,
        configurations: AskConfigurations | None = None,
    ):
        logger.info("SQL Generation pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "query": query,
                "documents": contexts,
                "exclude": exclude,
                "samples": samples,
                "project_id": project_id,
                "configurations": configurations,
                "previous_queries": previous_queries,
                **self._components,
                **self._configs,
            },
        )


if __name__ == "__main__":
    from langfuse.decorators import langfuse_context

    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.providers import init_providers
    from src.utils import init_langfuse, load_env_vars

    load_env_vars()
    init_langfuse()

    llm_provider, _, _, engine = init_providers(engine_config=EngineConfig())
    pipeline = SQLGeneration(
        llm_provider=llm_provider,
        engine=engine,
    )

    pipeline.visualize("this is a test query", [], [])
    async_validate(lambda: pipeline.run("this is a test query", [], []))

    langfuse_context.flush()
