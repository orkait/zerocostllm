from providers.scoring import score_models


def _m(**kw):
    base = dict(id="x", name="x", intel=None, intel_coding=None, intel_math=None, capability=10.0,
                tps=None, tools=False, brain=False, is_free=False, price_in=None, price_out=None,
                ctx=32768, uptime=None, provider="OpenRouter")
    base.update(kw)
    return base


def _by_id(scored):
    return {m["id"]: m for m in scored}


def test_empty_returns_empty():
    assert score_models([]) == []


def test_smarter_model_ranks_higher_overall():
    r = score_models([_m(id="smart", intel=55), _m(id="dumb", intel=10)])
    assert r[0]["id"] == "smart"
    assert _by_id(r)["smart"]["scores"]["intelligence"] > _by_id(r)["dumb"]["scores"]["intelligence"]


def test_free_model_gets_max_affordability():
    r = _by_id(score_models([_m(id="free", is_free=True, intel=30),
                             _m(id="paid", price_in=2e-6, price_out=1e-5, intel=30)]))
    assert r["free"]["scores"]["affordability"] == 100.0
    assert r["paid"]["scores"]["affordability"] < 100.0


def test_missing_intel_falls_back_to_capability_no_crash():
    # neither model has an AA intel index — must still score off capability, not crash
    r = score_models([_m(id="a", intel=None, capability=50), _m(id="b", intel=None, capability=5)])
    assert _by_id(r)["a"]["scores"]["intelligence"] > _by_id(r)["b"]["scores"]["intelligence"]


def test_no_tools_penalizes_tool_use():
    r = _by_id(score_models([_m(id="tooler", intel=50, tools=True),
                            _m(id="notool", intel=50, tools=False)]))
    assert r["tooler"]["scores"]["tool_use"] > r["notool"]["scores"]["tool_use"]
    assert r["notool"]["scores"]["tool_use"] < r["notool"]["scores"]["intelligence"]


def test_reasoning_ranks_by_measured_reasoning_benches():
    # reasoning is grounded in GPQA/HLE/long-context (measured), not the `brain` flag — a model that
    # actually scores high on reasoning benches wins, regardless of how it's tagged.
    strong = _m(id="strong", intel=40, brain=False)
    strong["benches"] = {"gpqa": 0.9, "hle": 0.5}
    weak = _m(id="weak", intel=40, brain=True)
    weak["benches"] = {"gpqa": 0.2, "hle": 0.1}
    r = _by_id(score_models([strong, weak]))
    assert r["strong"]["scores"]["reasoning"] > r["weak"]["scores"]["reasoning"]


def test_fit_code_favors_coding_specialist():
    r = _by_id(score_models([_m(id="coder", intel=40, intel_coding=58),
                            _m(id="generalist", intel=45, intel_coding=20)]))
    assert r["coder"]["scores"]["fit_code"] > r["generalist"]["scores"]["fit_code"]


def test_fit_budget_favors_cheap_and_decent():
    r = _by_id(score_models([_m(id="cheapsmart", is_free=True, intel=40),
                            _m(id="pricey", price_in=5e-6, price_out=2e-5, intel=45)]))
    assert r["cheapsmart"]["scores"]["fit_budget"] > r["pricey"]["scores"]["fit_budget"]


def test_fit_agent_favors_smart_tool_caller():
    r = _by_id(score_models([_m(id="agent", intel=50, tools=True, tps=200),
                            _m(id="notool", intel=55, tools=False, tps=200)]))
    assert r["agent"]["scores"]["fit_agent"] > r["notool"]["scores"]["fit_agent"]


def test_archetype_flagship_for_top_intelligence():
    r = _by_id(score_models([_m(id="top", intel=60), _m(id="mid", intel=5)]))
    assert r["top"]["archetype"] == "flagship"


def test_archetype_budget_for_free_smart():
    r = _by_id(score_models([_m(id="free", is_free=True, intel=50), _m(id="paid", price_in=9e-6, price_out=3e-5, intel=52)]))
    assert r["free"]["archetype"] in ("budget", "flagship")  # free+smart => high value


def test_badges_reflect_capabilities():
    r = _by_id(score_models([_m(id="f", is_free=True, tools=True, brain=True, intel=60, tps=700)]))
    b = r["f"]["badges"]
    assert "free" in b and "tools" in b and "reasoning" in b


def test_fast_free_provider_not_penalized_on_speed():
    # groq/cerebras report no tps but are the fastest inference lanes (LPU / wafer-scale). A zero-cost
    # tool must NOT rank them neutral-low just because throughput wasn't measured.
    r = _by_id(score_models([_m(id="g", provider="Groq", tps=None, intel=40),
                            _m(id="cb", provider="Cerebras", tps=None, intel=40),
                            _m(id="unknown", provider="MysteryCloud", tps=None, intel=40)]))
    assert r["g"]["scores"]["speed"] >= 80
    assert r["cb"]["scores"]["speed"] >= 80
    assert r["unknown"]["scores"]["speed"] == 50.0  # genuinely unknown → neutral, not penalized


def test_measured_tps_overrides_provider_prior():
    # if throughput IS measured, use it, not the prior
    r = _by_id(score_models([_m(id="slowgroq", provider="Groq", tps=10, intel=40),
                            _m(id="fastother", provider="X", tps=900, intel=40)]))
    assert r["fastother"]["scores"]["speed"] > r["slowgroq"]["scores"]["speed"]


def test_coding_grounded_in_real_benches():
    # coding = mean of livecodebench + scicode + terminalbench_hard (0..1 -> 0..100), not the aggregate
    m = _m(id="c", intel=45)
    m["benches"] = {"livecodebench": 0.8, "scicode": 0.6, "terminalbench_hard": 0.4}
    r = score_models([m])[0]
    assert abs(r["scores"]["coding"] - 60.0) < 0.1  # mean(80,60,40)
    assert r["bench_count"] == 3


def test_tool_use_from_agentic_benches_is_measured_not_guessed():
    # a MEASURED tau-bench / terminal-bench score replaces the boolean tools heuristic
    m = _m(id="measured", intel=30, tools=True)
    m["benches"] = {"tau2": 0.9, "terminalbench_hard": 0.7}
    r = score_models([m])[0]
    assert abs(r["scores"]["tool_use"] - 80.0) < 0.1  # mean(90,70)


def test_reasoning_grounded_in_gpqa_hle_lcr():
    m = _m(id="r", intel=30)
    m["benches"] = {"gpqa": 0.8, "hle": 0.4, "lcr": 0.6}
    r = score_models([m])[0]
    # gpqa/hle/lcr now all weight 0.9 (top-clean discriminative) → weighted mean = simple mean(80,40,60)
    assert abs(r["scores"]["reasoning"] - 60.0) < 0.3


def test_dynamic_bench_outweighs_stale_in_blend():
    # aime_25 (dynamic, weight 1.0) counts more than aime (older, weight 0.6) — a model strong on the
    # fresh bench and weak on the stale one scores above the naive mean.
    m = _m(id="m", intel=40)
    m["benches"] = {"aime_25": 0.9, "aime": 0.3}
    r = score_models([m])[0]
    assert r["scores"]["math"] > 60.5  # naive mean would be 60


def test_consensus_high_when_benches_agree():
    m = _m(id="a", intel=50)
    m["benches"] = {"gpqa": 0.8, "hle": 0.8, "livecodebench": 0.8, "tau2": 0.8}
    assert score_models([m])[0]["consensus"] >= 90


def test_consensus_low_when_benches_disagree():
    m = _m(id="d", intel=50)
    m["benches"] = {"gpqa": 0.9, "hle": 0.1, "livecodebench": 0.5, "tau2": 0.3}
    assert score_models([m])[0]["consensus"] < 50


def test_confidence_high_when_many_agreeing_benches():
    m = _m(id="s", intel=50)
    m["benches"] = {k: 0.7 for k in ["gpqa", "hle", "livecodebench", "scicode", "tau2", "lcr", "ifbench", "mmlu_pro"]}
    assert score_models([m])[0]["confidence"] == "high"


def test_confidence_low_when_thin_or_estimated():
    thin = _m(id="t", intel=50)
    thin["benches"] = {"gpqa": 0.7}
    assert score_models([thin])[0]["confidence"] == "low"
    est = _m(id="e", intel=None, capability=20, intel_est=True)
    assert score_models([est])[0]["confidence"] == "low"


def test_bench_blend_fail_open_on_partial():
    m = _m(id="p", intel=40)
    m["benches"] = {"livecodebench": 0.5}  # only one of the coding benches present
    r = score_models([m])[0]
    assert abs(r["scores"]["coding"] - 50.0) < 0.1  # mean of what's there


def test_falls_back_to_aggregate_index_then_intel():
    agg = _m(id="agg", intel=40, intel_coding=70)  # no granular benches, has AA coding_index (0..100)
    r = score_models([agg])[0]
    assert abs(r["scores"]["coding"] - 70.0) < 0.1  # used as-is, not divided by the intel ceiling
    bare = _m(id="bare", intel=40)
    r2 = score_models([bare])[0]
    assert r2["scores"]["coding"] == r2["scores"]["intelligence"]
    assert r2["bench_count"] == 0


def test_negative_price_sentinel_does_not_crash():
    # some providers use price -1 (BYOK/variable) — must not blow up log10 in the affordability math
    m = _m(id="byok", intel=40, price_in=-1.0, price_out=-1.0)
    r = score_models([m, _m(id="ok", intel=40, price_in=2e-6, price_out=1e-5)])[0]
    assert 0 <= r["scores"]["affordability"] <= 100


def test_totally_sparse_model_does_not_crash():
    # a model with only id/name and everything else None/default
    r = score_models([{"id": "sparse", "name": "sparse"}])
    assert len(r) == 1
    assert 0 <= r[0]["scores"]["overall"] <= 100


def test_uptime_as_percentage_does_not_explode_overall():
    # OpenRouter sends uptime as a percentage (e.g. 99.2), not a 0..1 fraction — overall must stay <=100
    m = _m(id="up", intel=50)
    m["uptime"] = 99.2
    r = score_models([m])[0]
    assert 0 <= r["scores"]["reliability"] <= 100
    assert 0 <= r["scores"]["overall"] <= 100


def test_the_most_expensive_model_scores_zero_affordability_not_neutral_fifty():
    """`_norm(...) or 50.0` swallowed a legitimate 0.0.

    The cheapest model in a pool normalizes to 100 and the dearest to exactly 0 - and 0 is falsy, so
    the worst-priced model in the market was quietly promoted to 50, the value that means "we do not
    know this model's price". The two are opposite claims and must not share a number.
    """
    r = _by_id(score_models([
        _m(id="cheap", price_in=1e-6, price_out=1e-6, intel=30),
        _m(id="dear", price_in=1e-3, price_out=1e-3, intel=30),
    ]))
    assert r["dear"]["scores"]["affordability"] == 0.0
    assert r["cheap"]["scores"]["affordability"] == 100.0


def test_an_unknown_price_is_still_the_cautious_neutral():
    """The fallback itself is unchanged - only reaching it by accident was the bug."""
    r = _by_id(score_models([_m(id="unpriced", price_in=None, price_out=None, intel=30)]))
    assert r["unpriced"]["scores"]["affordability"] == 60.0
    assert r["unpriced"]["scores"]["value"] == 50.0


def test_worst_value_in_the_pool_is_also_not_promoted_to_neutral():
    r = _by_id(score_models([
        _m(id="great", price_in=1e-9, price_out=1e-9, intel=60),
        _m(id="awful", price_in=1e-2, price_out=1e-2, intel=5),
    ]))
    assert r["awful"]["scores"]["value"] == 0.0
    assert r["great"]["scores"]["value"] == 100.0
