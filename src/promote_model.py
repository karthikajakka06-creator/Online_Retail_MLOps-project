import mlflow
from mlflow import MlflowClient

from tracking_utils import REGISTERED_MODEL_NAME, configure_mlflow


def main() -> None:
    configure_mlflow()
    client = MlflowClient()
    candidate = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate")
    candidate_f1 = float(client.get_run(candidate.run_id).data.metrics["f1"])

    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        champion = None

    champion_f1 = None
    if champion is not None:
        champion_f1 = float(client.get_run(champion.run_id).data.metrics["f1"])
    promoted = champion_f1 is None or candidate_f1 >= champion_f1
    if promoted:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias="champion", version=candidate.version
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=candidate.version,
            key="lifecycle_status",
            value="champion",
        )
        print(f"Promoted version {candidate.version} (F1={candidate_f1:.4f}) to champion.")
    else:
        print(
            f"Kept existing champion (F1={champion_f1:.4f}); "
            f"candidate F1={candidate_f1:.4f}."
        )


if __name__ == "__main__":
    main()