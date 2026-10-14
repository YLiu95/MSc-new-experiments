import pytest
import torch

from ranking17.warmstart import merge_parameter


@pytest.mark.parametrize("components,name", [(3, "temporal.0.qkv_weight"), (2, "film_weight")])
def test_packed_components_merge_in_component_order(components, name):
    original = torch.arange(components * 16 * 5).reshape(components * 16, 5)
    component_chunks = [component.chunk(8, dim=0) for component in original.chunk(components, dim=0)]
    shards = [torch.cat([component[rank] for component in component_chunks]) for rank in range(8)]
    for rank in range(4):
        expected = torch.cat([component.chunk(4, dim=0)[rank] for component in original.chunk(components, dim=0)])
        assert torch.equal(merge_parameter(shards[2 * rank:2 * rank + 2], "qkv", name), expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_regular_shards_merge_on_partition_axis(axis):
    original = torch.arange(32 * 32).reshape(32, 32)
    shards = original.chunk(8, dim=axis)
    for rank in range(4):
        assert torch.equal(merge_parameter(shards[rank * 2:rank * 2 + 2], axis, "weight"),
                           original.chunk(4, dim=axis)[rank])


def test_replicated_parameters_must_agree():
    values = torch.arange(16)
    assert torch.equal(merge_parameter([values, values.clone()], None, "norm"), values)
    with pytest.raises(ValueError, match="Replicated"):
        merge_parameter([values, values + 1], None, "norm")