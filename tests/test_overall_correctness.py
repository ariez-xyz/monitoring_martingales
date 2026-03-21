from random import random
from monitor.adapters.interface import DynamicalSystemAdapter

def test_lipschitz(adapter: DynamicalSystemAdapter):
    B = adapter.get_increment_bound()
    seed = int(random()*999999)
    adapter.reset(seed=seed)
    last_certificate_value = adapter.get_certificate_value()
    while not adapter.done():
        adapter.step()
        delta = last_certificate_value - adapter.get_certificate_value()
        last_certificate_value = adapter.get_certificate_value()
        assert delta < B, f"Lipschitz bound fails for adapter {adapter}, seed {seed}"
