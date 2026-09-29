from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from ranker.backup import local
from ranker.checkpoints import digest, pointer
from ranker.contract import DATASET, REVISION
from ranker.data import PREPARATION_POLICY, write_json
from ranker.publish import GITHUB_FOLDER, GITHUB_REPO, credential, git_blob_file


REPOSITORIES = {
    "A": "YL95/experiment-1.6-asc-gpu-nodes-arm-a",
    "B": "YL95/experiment-1.6-asc-gpu-nodes-arm-b",
    "C": "YL95/experiment-1.6-asc-gpu-nodes",
}


def files_for_arm(root: Path, arm: str, source: dict) -> tuple[dict[str, Path], bool]:
    if arm not in REPOSITORIES or source.get("repository") != GITHUB_REPO or not source.get("private"):
        raise ValueError("A verified private source commit and registered arm are required")
    rights = json.loads((root / "control" / "publication_authorization.json").read_text())
    if rights.get("confirmed_by_user") is not True or rights.get("training_root") != str(root.resolve()):
        raise ValueError("Explicit user authorization for this public artifact root is missing")
    meta = json.loads((root / "panel" / "meta.json").read_text())
    if meta["revision"] != REVISION or meta.get("preparation_policy") != PREPARATION_POLICY:
        raise ValueError("Only the imputation-masked pinned dataset may be published")
    selected = root / arm
    finished = json.loads((selected / "TRAINING_DONE.json").read_text())
    best, latest = pointer(selected, "best"), pointer(selected, "latest")
    best_state = json.loads((best / "state.json").read_text())
    latest_state = json.loads((latest / "state.json").read_text())
    if not 0 <= best_state["step"] <= latest_state["step"] == finished["step"] or finished["step"] < 1:
        raise ValueError("A completed screen with consistent best/latest states is required")
    config = json.loads((selected / "config.json").read_text())
    registry = json.loads((root / "panel" / "evaluation" / "val-primary.json").read_text())
    if (config["data"]["preparation_policy"] != PREPARATION_POLICY
            or config["validation"]["primary"] != registry["sha256"] or config["arm"] != arm):
        raise ValueError("Ranking checkpoint is incompatible with the corrected validation/data policy")
    summary = json.loads((selected / "reports" / "training_summary.json").read_text())
    if not summary.get("full_validation_selection"):
        raise ValueError("Best checkpoint was not selected on the complete primary suite")
    publication = root / "publication" / arm
    publication.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = {f"best/{name}": best / name for name in ("weights.safetensors", "config.json", "state.json")}
    write_json(publication / "inference_manifest.json", {
        name: {"bytes": paths[f"best/{name}"].stat().st_size, "sha256": digest(paths[f"best/{name}"])}
        for name in ("weights.safetensors", "config.json", "state.json")})
    paths["best/inference_manifest.json"] = publication / "inference_manifest.json"
    same_checkpoint = best.resolve() == latest.resolve()
    if not same_checkpoint:
        paths.update({f"latest/{path.name}": path for path in latest.iterdir() if path.is_file()})
        if len(list(latest.glob("rng-*.pt"))) != latest_state["world"]:
            raise ValueError("Latest recovery lacks per-rank RNG states")
    paths["vocabulary.json"] = root / "panel" / "vocabulary.json"
    for name in ("baselines.json", "evaluation_registry.json"):
        paths[f"reports/{name}"] = root / "reports" / name
    paths["reports/training_summary.json"] = selected / "reports" / "training_summary.json"
    paths["reports/validation.jsonl"] = selected / "validation.jsonl"
    paths["reports/history.jsonl"] = selected / "history.jsonl"
    for path in sorted((selected / "runs").rglob("events.out.tfevents.*")):
        paths[f"runs/{path.relative_to(selected / 'runs').as_posix()}"] = path
    if not any(name.startswith("runs/") for name in paths):
        raise ValueError(f"No aggregate TensorBoard event files for arm {arm}")
    link = (f"https://github.com/{GITHUB_REPO}/blob/{source['commit']}/"
            f"{GITHUB_FOLDER}/MODEL_CARD.md")
    (publication / "README.md").write_text(f"[Private model card]({link})\n")
    write_json(publication / "reconstruction.json", {
        "dataset": DATASET, "revision": REVISION, "preparation_policy": PREPARATION_POLICY,
        "return_scale_pct": meta["return_scale_pct"], "input_clip": meta["input_clip"],
        "training_identities": meta["n_tickers"], "arm": arm,
        "best_step": best_state["step"], "best_at_initialization": best_state["step"] == 0,
        "latest_step": latest_state["step"], "latest_same_as_best": same_checkpoint,
        "public_recovery_included": not same_checkpoint,
        "source_commit": source["commit"], "vocabulary_sha256": digest(paths["vocabulary.json"]),
        "source_card_attribution_disputed_by_data_owner": True})
    paths["README.md"] = publication / "README.md"
    paths["reconstruction.json"] = publication / "reconstruction.json"
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("A selected model, aggregate report, or reconstruction file is missing")
    return paths, same_checkpoint


def publish_arm(root: Path, arm: str, source: dict, api: HfApi) -> dict:
    paths, same_checkpoint = files_for_arm(root, arm, source)
    repository = REPOSITORIES[arm]
    api.create_repo(repo_id=repository, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repository).private:
        raise ValueError("The user's requested model destination must be public")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in paths.items()]
    previous = root / "reports" / f"public_backup_verification_{arm}.json"
    if previous.is_file():
        old = json.loads(previous.read_text())
        if old.get("repository") == repository:
            operations.extend(CommitOperationDelete(path_in_repo=name)
                              for name in old["verified_files"] if name not in paths)
    commit = api.create_commit(repo_id=repository, repo_type="model", operations=operations,
                               commit_message=f"Back up Experiment 1.6 arm {arm} screen best and distinct latest", num_threads=8)
    remote = {entry.rfilename: entry for entry in api.model_info(repository, revision=commit.oid,
                                                                  files_metadata=True).siblings}
    verified = {}
    for name, path in paths.items():
        entry = remote.get(name)
        if entry is None or entry.size != path.stat().st_size:
            raise ValueError(f"Remote file size mismatch in {repository}: {name}")
        expected = digest(path) if entry.lfs is not None else git_blob_file(path)
        actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        if actual != expected:
            raise ValueError(f"Remote file digest mismatch in {repository}: {name}")
        verified[name] = {"bytes": entry.size, "digest": actual}
    result = {"repository": repository, "commit": commit.oid, "arm": arm,
              "best_equals_latest": same_checkpoint, "verified_files": verified,
              "verified_at": datetime.now().astimezone().isoformat()}
    write_json(previous, result)
    print(json.dumps({"event": "public_backup_verified", "repo": repository, "arm": arm,
                      "files": len(verified), "bytes": sum(item["bytes"] for item in verified.values()),
                      "commit": commit.oid}), flush=True)
    return result


def publish_all(root: Path, screen_job_id: str) -> dict:
    root = root.resolve()
    source = json.loads((root / "control" / "github_source.json").read_text())
    if source.get("repository") != GITHUB_REPO or not source.get("private"):
        raise ValueError("Verify the private GitHub source commit before any public upload")
    reports = local(root, screen_job_id, all_arms=True)
    if set(reports) != set(REPOSITORIES) or any(report["status"] != "verified" for report in reports.values()):
        raise ValueError("All three independent private backups must be verified before public upload")
    api = HfApi(token=credential("HF_TOKEN"))
    if api.whoami().get("name") != "YL95":
        raise ValueError("HF credentials do not own the three requested model repositories")
    dataset = api.dataset_info(DATASET, revision=REVISION)
    if dataset.id != DATASET:
        raise ValueError("Pinned dataset provenance changed")
    results = {}
    for arm in REPOSITORIES:
        results[arm] = publish_arm(root, arm, source, api)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-job-id", required=True)
    arguments = parser.parse_args()
    publish_all(arguments.root, arguments.screen_job_id)


if __name__ == "__main__":
    main()