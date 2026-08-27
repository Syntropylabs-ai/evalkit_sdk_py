from evalkit.core.otel_provider import _readable_to_envelope


class _Ctx:
    trace_id = 0x1
    span_id = 0x2
    is_valid = True


class _Span:

    def __init__(self, attrs):
        self._a = attrs
        self.name = "chat"
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None
        self.parent = None
        self.instrumentation_scope = None

    def get_span_context(self):
        return _Ctx()

    @property
    def attributes(self):
        return self._a


def test_upstream_cache_spellings_reach_typed_fields(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-sonnet-4-5",
        "gen_ai.usage.input_tokens": 120_000,
        "gen_ai.usage.output_tokens": 300,
        "gen_ai.usage.cache_read.input_tokens": 118_000,
        "gen_ai.usage.cache_creation.input_tokens": 1_500,
    }))
    assert env.cache_read_tokens == 118_000, (
        "the mapper accepted only EvalKit's own gen_ai.usage.cache_read_tokens spelling, so the upstream semconv "
        "name gen_ai.usage.cache_read.input_tokens landed in no typed field; the tracing dashboard's "
        "estimateSpanCost subtracts cacheReadTokens out of tokensIn and reprices them, so a dropped cache bucket "
        "bills all 118k cached tokens at the full input rate"
    )
    assert env.cache_creation_tokens == 1_500, (
        "gen_ai.usage.cache_creation.input_tokens was dropped too; cache writes are the most expensive input "
        "bucket there is, and the cost estimate can only reprice a bucket it was told about"
    )


def test_deprecated_token_spellings_are_read(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.prompt_tokens": 500,
        "gen_ai.usage.completion_tokens": 40,
    }))
    assert env.tokens_in == 500 and env.tokens_out == 40, (
        "a span using the deprecated gen_ai.usage.prompt_tokens/completion_tokens spellings arrived with tokens_in "
        "None, and TraceEnvelope.to_dict omits tokensIn entirely when it is None, so the span stores 0 tokens and "
        "prices at exactly $0 - indistinguishable on the dashboard from a call that cost nothing"
    )


def test_modern_spelling_wins_over_the_deprecated_one(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": 742,
        "gen_ai.usage.prompt_tokens": 742,
    }))
    assert env.tokens_in == 742, (
        "an instrumentation that dual-emits both spellings of the same number must be read once; summing or "
        "preferring the deprecated key would double the priced input tokens of every span it produces"
    )


def test_provider_falls_back_to_provider_name(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.model": "claude-sonnet-4-5",
        "gen_ai.usage.input_tokens": 10,
    }))
    assert env.provider == "anthropic", (
        "gen_ai.system is deprecated upstream in favour of gen_ai.provider.name, so a span from a current-semconv "
        "instrumentation arrived with provider NULL and cannot be grouped, filtered or attributed by provider "
        "anywhere in the product"
    )
    assert env.span_type == "llm_call", (
        "a span whose only provider marker is gen_ai.provider.name was typed function_call, which drops it out of "
        "every llm_call-gated token and cost rollup in trace-service2"
    )


def test_provider_name_alone_types_the_span_as_llm_call(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.provider.name": "openai",
        "gen_ai.usage.input_tokens": 80,
        "gen_ai.usage.output_tokens": 5,
    }))
    assert env.span_type == "llm_call", (
        "gen_ai.provider.name has to type the span on its own, not merely alongside a model attribute: a stream that "
        "fails before the response arrives carries usage and a provider but no gen_ai.response.model, and typing it "
        "function_call drops real tokens out of every llm_call-gated rollup in trace-service2"
    )


def test_gen_ai_system_wins_over_provider_name(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "bedrock",
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.usage.input_tokens": 10,
    }))
    assert env.provider == "bedrock", (
        "an instrumentation that emits both keys spells them differently (gen_ai.system=bedrock vs "
        "gen_ai.provider.name=aws.bedrock); picking the new key would silently rename an existing provider and "
        "split its history into two series on every dashboard that groups by provider"
    )


def test_token_counts_are_coerced_to_int(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "gemini",
        "gen_ai.usage.input_tokens": 1135.0,
        "gen_ai.usage.output_tokens": "20",
    }))
    assert type(env.tokens_in) is int and env.tokens_in == 1135, (
        "an OTel token attribute that is a float was forwarded verbatim, so json.dumps wrote tokensIn as 1135.0; "
        "trace-service2 decodes TokensIn as a Go int and json.Unmarshal rejects 1135.0, and the decode covers the "
        "whole BatchIngestRequest, so one such span 400s the entire batch - which sdk-py's exporter treats as "
        "terminal and never retries, losing every span batched with it"
    )
    assert type(env.tokens_out) is int and env.tokens_out == 20, (
        "a numeric-string token count reached the wire as a JSON string; the same Go decode rejects it and takes "
        "the whole batch down with it"
    )


def test_non_finite_token_count_does_not_drop_the_span(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": float("inf"),
        "gen_ai.usage.output_tokens": float("nan"),
        "gen_ai.usage.prompt_tokens": 12,
    }))
    assert env.tokens_in == 12, (
        "int(float('inf')) raises OverflowError, not ValueError; an unguarded coercion propagates out of the "
        "mapper into the exporter's blanket except, which logs at debug and drops the whole span - and a span "
        "never emitted is indistinguishable from a call that never happened, which is strictly worse than the "
        "wrong token count it was trying to avoid"
    )
    assert env.tokens_out is None, (
        "nan must be rejected rather than coerced; json.dumps writes NaN, which is not valid JSON and is rejected "
        "by the Go decoder for the entire batch"
    )


def test_zero_never_shadows_a_real_cache_value(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "anthropic",
        "gen_ai.usage.input_tokens": 1_000,
        "gen_ai.usage.cache_read_tokens": 0,
        "gen_ai.usage.cache_read.input_tokens": 900,
    }))
    assert env.cache_read_tokens == 900, (
        "a provider that reports cache_read_tokens=0 alongside a real gen_ai.usage.cache_read.input_tokens must "
        "not win the lookup just because its spelling is checked first; storing 0 there hands the whole cached "
        "prompt back to the full-price input bucket"
    )


def test_absent_cache_stays_none_rather_than_zero(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": 30,
    }))
    assert env.cache_read_tokens is None and env.cache_creation_tokens is None, (
        "an absent cache bucket must stay None so to_dict omits the key; emitting an explicit 0 would be a claim "
        "about the provider's caching that the span never made"
    )


def test_exclusive_cache_convention_is_folded_into_tokens_in(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.cache_read.input_tokens": 40_000,
        "gen_ai.usage.cache_creation.input_tokens": 200,
    }))
    assert env.tokens_in == 40_212, (
        "Anthropic-native and Bedrock report input_tokens EXCLUSIVE of the cache buckets, while the dashboard "
        "computes full-price input as max(0, tokensIn - cacheRead - cacheCreation) and therefore treats tokensIn "
        "as inclusive; passing an exclusive 12 straight through collapses the full-price bucket to 0 and "
        "undercounts the call. tokens_in < cacheRead + cacheCreation is only possible under the exclusive "
        "convention, so it is a safe signal to fold on"
    )


def test_inclusive_cache_convention_is_left_alone(monkeypatch):
    monkeypatch.setenv("EVALKIT_COUNT_FOREIGN_LLM", "1")
    env = _readable_to_envelope(_Span({
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": 40_212,
        "gen_ai.usage.cached_tokens": 40_000,
    }))
    assert env.tokens_in == 40_212, (
        "OpenAI and Gemini already include cached tokens in input_tokens; folding again would double-count 40k "
        "tokens on every cache hit, which is the same money bug in the opposite direction"
    )


def test_deprecated_only_foreign_span_is_still_deferred(monkeypatch):
    monkeypatch.delenv("EVALKIT_COUNT_FOREIGN_LLM", raising=False)
    env = _readable_to_envelope(_Span({
        "gen_ai.provider.name": "openai",
        "gen_ai.usage.prompt_tokens": 900,
        "gen_ai.usage.completion_tokens": 12,
        "gen_ai.usage.cache_read.input_tokens": 700,
    }))
    assert env.tokens_in is None and env.tokens_out is None, (
        "widening the read side without widening the foreign-span guard makes a deprecated-spelling span newly "
        "countable while it escapes the dedupe check, and evalkit.traces is a plain MergeTree written with INSERT "
        "FORMAT JSONEachRow - a span counted twice there permanently double-counts both its tokens and its cost"
    )
    assert env.cache_read_tokens is None and env.cache_creation_tokens is None, (
        "the cache buckets are cost inputs like the token counts and must be deferred with them; leaking a cache "
        "bucket onto a span whose tokens were deliberately withheld produces a negative full-price bucket"
    )
    assert env.span_type == "function_call", (
        "a deferred foreign LLM span must not be typed llm_call, or it joins the llm_call-filtered rollups with "
        "zero tokens and drags the average cost per call down"
    )
