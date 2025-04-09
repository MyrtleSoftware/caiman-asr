from cerberus import Validator

data_schema = {
    "datasets": {
        "type": "dict",
        "keysrules": {"type": "string"},
        "valuesrules": {
            "type": "dict",
            "schema": {
                "manifest": {"type": "string", "required": True},
                "weight": {"type": "float", "min": 0, "default": 1.0},
            },
        },
    }
}


class DatasetSchemaValidator:
    def __init__(self):
        self.validator = Validator(
            data_schema, purge_unknown=True
        )  # Remove unexpected keys

    def validate(self, data: dict) -> dict:
        """Validate and normalize dataset YAML contents."""
        if not self.validator.validate(data):
            raise ValueError(f"Invalid YAML format: {self.validator.errors}")
        return self.validator.document
