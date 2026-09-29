from pathlib import Path

import pytest

from gearbox.config import GearboxConfig, Pricing, Tier, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_loads():
    config = load_config(ROOT / "gearbox.example.yaml")
    assert [t.name for t in config.tiers] == ["nano", "small", "medium", "large"]
    assert config.host.cache_read == 0.3
    assert config.tier_index("medium") == 2
    assert config.top == 3


def test_pricing_cost_uses_cache_rate():
    p = Pricing(input=10.0, output=50.0, cache_read=1.0)
    # 600 uncached input + 400 cached input + 100 output
    assert p.cost(1000, 100, cached_tokens=400) == pytest.approx((600 * 10 + 400 * 1 + 100 * 50) / 1e6)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tiers": ()}, "at least one tier"),
        ({"tiers": (Tier("a", "m"), Tier("a", "m"))}, "unique"),
        ({"tiers": (Tier("a", "m"),), "leverage": -1}, "leverage"),
        ({"tiers": (Tier("a", "m"),), "difficulty": "vibes"}, "estimator"),
    ],
)
def test_invalid_configs_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GearboxConfig(**kwargs)


def test_unknown_tier_name():
    config = GearboxConfig(tiers=(Tier("a", "m"),))
    with pytest.raises(KeyError):
        config.tier_index("b")
