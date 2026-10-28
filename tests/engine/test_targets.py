"""A run whose custom targets have no patterns is refused when it's sent, not after its model
has loaded ("no layers matched network.targets=[]")."""

import pytest

from zentrainer.spec import SpecError, TrainSpec


def _spec(**network):
    s = TrainSpec().to_dict()
    s["model"].update(ckpt_name="x.safetensors")
    s["dataset"]["folder"] = "d"
    s["network"].update(network)
    return s


def test_custom_targets_need_a_pattern():
    for empty in ([], [""], ["  "]):
        with pytest.raises(SpecError, match="no patterns"):
            TrainSpec.from_dict(_spec(targets=empty)).validate()
    TrainSpec.from_dict(_spec(targets=[r"\.attn\."])).validate()
    TrainSpec.from_dict(_spec(targets="mlp")).validate()


def test_an_empty_list_still_trains_biases_and_norms():
    TrainSpec.from_dict(_spec(targets=[], train_bias_norm=True)).validate()
