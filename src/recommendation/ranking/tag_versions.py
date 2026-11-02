from mlflow import MlflowClient

from .config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)

for version in ["1", "2", "3"]:
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="validation_status",
        value="invalid",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="reason",
        value="target_leakage",
    )

print("Tagged v1-v3 as invalid.")

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="candidate",
    version="4",
)

print("candidate -> v4")
print("Done.")