from caiman_asr_train.utils.num_weights import num_weights


def test_model_size(mini_model_factory):
    model, _ = mini_model_factory()
    assert num_weights(model) == 1538
