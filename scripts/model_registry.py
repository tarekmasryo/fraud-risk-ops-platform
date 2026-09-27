"""Inspect local model versions and explicitly promote a reviewed candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAMES = ("fraud-risk-rf", "fraud-risk-xgb")


def _version_details(version) -> dict:
    return {
        "name": version.name,
        "version": str(version.version),
        "run_id": version.run_id,
        "status": version.status,
        "tags": version.tags,
    }


def list_versions(client: MlflowClient, name: str) -> list[dict]:
    versions = []
    token = None
    while True:
        page = client.search_model_versions(f"name = '{name}'", page_token=token)
        versions.extend(_version_details(version) for version in page)
        token = page.token
        if not token:
            return sorted(versions, key=lambda version: int(version["version"]))


def set_candidate(client: MlflowClient, name: str, version: str) -> dict:
    selected = client.get_model_version(name, version)
    if selected.status != "READY":
        raise ValueError(f"Model version {name}/{version} is not READY.")
    client.set_registered_model_alias(name, "candidate", selected.version)
    return {"alias": "candidate", **_version_details(selected)}


def promote_candidate(client: MlflowClient, name: str) -> dict:
    candidate = client.get_model_version_by_alias(name, "candidate")
    if candidate.status != "READY":
        raise ValueError(f"Candidate {name}/{candidate.version} is not READY.")
    client.set_registered_model_alias(name, "champion", candidate.version)
    return {"alias": "champion", **_version_details(candidate)}


def smoke_champion(client: MlflowClient, name: str, input_csv: Path) -> dict:
    champion = client.get_model_version_by_alias(name, "champion")
    expected = mlflow.models.get_model_info(f"models:/{name}/{champion.version}")
    uri = f"models:/{name}@champion"
    model = mlflow.pyfunc.load_model(uri)
    if model.metadata.model_uuid != expected.model_uuid:
        raise ValueError("Champion changed while loading; retry against the reviewed alias.")
    frame = pd.read_csv(input_csv)
    if frame.empty:
        raise ValueError("Smoke input must contain at least one row of model features.")
    predictions = model.predict(frame)
    return {
        "model_uri": uri,
        "version": str(champion.version),
        "run_id": champion.run_id,
        "rows": len(frame),
        "probabilities": predictions.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db",
        help="Tracking and registry store (default: MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
    )
    parser.add_argument("--name", choices=MODEL_NAMES, default=MODEL_NAMES[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List versions and their evaluation/provenance tags.")
    commands.add_parser("aliases", help="Inspect registered model aliases.")
    candidate = commands.add_parser("set-candidate", help="Point candidate to a selected version.")
    candidate.add_argument("--version", required=True)
    commands.add_parser("promote", help="Explicitly point champion to the current candidate.")
    commands.add_parser("champion", help="Show the version currently behind champion.")
    smoke = commands.add_parser("smoke", help="Load champion by alias and predict probabilities.")
    smoke.add_argument("--input-csv", type=Path, required=True, help="CSV of compatible features.")
    args = parser.parse_args(argv)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    try:
        if args.command == "list":
            result = list_versions(client, args.name)
        elif args.command == "aliases":
            result = {
                alias: str(version)
                for alias, version in client.get_registered_model(args.name).aliases.items()
            }
        elif args.command == "set-candidate":
            result = set_candidate(client, args.name, args.version)
        elif args.command == "promote":
            result = promote_candidate(client, args.name)
        elif args.command == "champion":
            result = _version_details(client.get_model_version_by_alias(args.name, "champion"))
        else:
            result = smoke_champion(client, args.name, args.input_csv)
    except (MlflowException, ValueError, OSError) as exc:
        parser.exit(1, f"Registry operation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
