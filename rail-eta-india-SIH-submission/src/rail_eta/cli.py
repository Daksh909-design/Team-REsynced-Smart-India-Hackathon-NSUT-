"""Reproducible command-line workflow for collection, training, and serving."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import uvicorn

from rail_eta.api import create_app
from rail_eta.bootstrap import load_public_delay_table, save_bootstrap, train_bootstrap
from rail_eta.config import Settings
from rail_eta.connectors.railradar import collect_once
from rail_eta.supervised import build_training_frame, load_training_frame, save_training_frame
from rail_eta.training import save_bundle, train


def _numbers(value: str) -> list[str]:
    numbers = [item.strip() for item in value.split(",") if item.strip()]
    if not numbers:
        raise argparse.ArgumentTypeError("provide at least one comma-separated train number")
    return numbers


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="rail-eta", description="Indian Railways ETA model workflow"
    )
    sub = root.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="capture immutable real live-status snapshots")
    collect.add_argument(
        "--trains", type=_numbers, required=True, help="comma-separated 5-digit train numbers"
    )
    collect.add_argument(
        "--journey-date", help="YYYY-MM-DD; omit to let provider select the active run"
    )
    collect.add_argument("--snapshot-dir", type=Path)
    collect.add_argument("--concurrency", type=int, default=4)

    build = sub.add_parser("build", help="turn snapshots into point-in-time labelled examples")
    build.add_argument("--snapshot-dir", type=Path, default=Path("data/raw/snapshots"))
    build.add_argument("--output", type=Path, default=Path("data/processed/training_examples.csv"))
    build.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))

    fit = sub.add_parser("train", help="temporally validate and train the model")
    fit.add_argument("--data", type=Path, default=Path("data/processed/training_examples.csv"))
    fit.add_argument("--model", type=Path, default=Path("models/eta_model.joblib"))
    fit.add_argument("--report", type=Path, default=Path("reports/model_report.json"))

    serve = sub.add_parser("serve", help="start the dynamic ETA API")
    serve.add_argument("--model", type=Path, default=Path("models/eta_model.joblib"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    bootstrap = sub.add_parser(
        "bootstrap", help="train a cold-start prior from a public station-level delay CSV"
    )
    bootstrap.add_argument("--csv", type=Path, required=True)
    bootstrap.add_argument("--model", type=Path, default=Path("models/bootstrap_prior.joblib"))
    bootstrap.add_argument("--report", type=Path, default=Path("reports/bootstrap_report.json"))
    return root


def main() -> None:
    args = parser().parse_args()
    settings = Settings.from_env()
    if args.command == "collect":
        snapshot_dir = args.snapshot_dir or settings.snapshot_dir
        written, failures = asyncio.run(
            collect_once(
                args.trains,
                api_key=settings.railradar_api_key,
                base_url=settings.railradar_base_url,
                snapshot_dir=snapshot_dir,
                journey_date=args.journey_date,
                concurrency=args.concurrency,
                weather_enabled=settings.weather_enabled,
            )
        )
        print(
            json.dumps({"written": [str(path) for path in written], "failures": failures}, indent=2)
        )
        return
    if args.command == "build":
        frame, audit = build_training_frame(args.snapshot_dir)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        if frame.empty:
            raise SystemExit(
                "No labelled examples yet. Continue collecting until completed journeys are observed."
            )
        save_training_frame(frame, args.output)
        print(json.dumps({**audit, "output": str(args.output)}, indent=2))
        return
    if args.command == "train":
        bundle, report = train(load_training_frame(args.data))
        save_bundle(bundle, report, args.model, args.report)
        print(
            json.dumps({"selected_model": bundle.model_name, "report": str(args.report)}, indent=2)
        )
        return
    if args.command == "serve":
        uvicorn.run(create_app(args.model, settings), host=args.host, port=args.port)
        return
    if args.command == "bootstrap":
        bundle, report = train_bootstrap(load_public_delay_table(args.csv))
        save_bootstrap(bundle, report, args.model, args.report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
