import json
from enum import Enum


class CheckpointNotSupportedError(Exception):
    pass


class ModelVariant(Enum):
    TESTING = "testing"
    BASE = "base"
    LARGE = "large"


def return_schemas():
    schemas = []
    for model in ModelVariant:
        with open(
            f"/workspace/training/caiman_asr_train/export/model_schema/{model.value}.json",
            "r",
        ) as file:
            schema = json.load(file)
            schemas.append(schema)
    return schemas


def check_model_schema(model_sd, schemas):
    """
    Raise CheckpointNotSupportedError if state dict isn't supported.

    This is necessary to avoid incompatibility with downstream inference server.
    """
    model_schema = {k: list(v.shape) for (k, v) in model_sd.items()}
    matching_schema = [schema for schema in schemas if schema == model_schema]
    if len(matching_schema) != 1:
        raise CheckpointNotSupportedError(
            "Model checkpoint's state dict sizes does not match any of the supported "
            f"ModelVariant options={[x.name for x in ModelVariant]}."
        )


def check_schema_training(model_sd: dict, skip_state_dict_check: bool):
    """
    Check state dict matches one of ModelVariant schemas.

    Raise CheckpointNotSupportedError if state dict isn't supported except when
    --skip_state_dict_check is passed.
    """
    try:
        check_model_schema(model_sd, return_schemas())
    except CheckpointNotSupportedError as e:
        if not skip_state_dict_check:
            # Extend error message
            err_msg = str(e)
            err_msg += (
                "\nIf you would like to avoid this check, pass --skip_state_dict_check. "
                "NOTE that skipping this check will make your model incompatible with "
                "the Myrtle.ai inference server."
            )
            raise CheckpointNotSupportedError(err_msg)
