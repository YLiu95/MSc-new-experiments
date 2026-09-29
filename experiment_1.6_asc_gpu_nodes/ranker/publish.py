from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi
import requests

from .checkpoints import digest, pointer
from .contract import DATASET, REVISION
from .data import write_json


GITHUB_REPO = "YLiu95/MSc-new-experiments"
GITHUB_FOLDER = "experiment_1.6_asc_gpu_nodes"
HF_REPO = "YL95/experiment-1.6-asc-gpu-nodes"


def credential(name: str) -> str:
    location = Path.home() / ".env"
    if not location.is_file() or stat.S_IMODE(location.stat().st_mode) & 0o077:
        raise ValueError("The private dotenv file must exist with owner-only permissions")
    load_dotenv(location, override=False)
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required {name} credential is unavailable")
    return value


def github_files(source: Path, plan: Path) -> dict[str, bytes]:
    included = {}
    ignored = {".pytest_cache", "__pycache__", "artifacts", "cache", "logs", ".git"}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ignored.intersection(path.relative_to(source).parts):
            continue
        if path.name != ".gitignore" and path.suffix not in (".py", ".sh", ".md", ".txt", ".json", ".docx"):
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Source file is unexpectedly large: {path.name}")
        included[f"{GITHUB_FOLDER}/{path.relative_to(source).as_posix()}"] = path.read_bytes()
    if not plan.is_file():
        raise ValueError("The registered research plan is missing")
    included[f"{GITHUB_FOLDER}/EXPERIMENT_PLAN.md"] = plan.read_bytes()
    if f"{GITHUB_FOLDER}/MODEL_CARD.md" not in included:
        raise ValueError("The full private model card must be included in the source commit")
    if any(part in name.lower() for name in included for part in ("token", ".env", ".npy", ".pt")):
        raise ValueError("Private data or credentials escaped the source allowlist")
    return included


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def github_commit(source: Path, plan: Path, root: Path) -> dict:
    files = github_files(source, plan)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {credential('GITHUB_TOKEN')}",
                            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})

    def api(method: str, path: str, **arguments):
        response = session.request(method, f"https://api.github.com/repos/{GITHUB_REPO}{path}",
                                   timeout=45, **arguments)
        response.raise_for_status()
        return response.json()

    repository = api("GET", "")
    if not repository["private"] or not repository.get("permissions", {}).get("push"):
        raise ValueError("Expected a writable private source repository")
    branch = repository["default_branch"]
    if repository["size"] == 0:
        first = f"{GITHUB_FOLDER}/README.md"
        api("PUT", f"/contents/{first}", json={"message": "Initialize private Experiment 1.6 source",
                                               "content": base64.b64encode(files[first]).decode(), "branch": branch})
    head = api("GET", f"/git/ref/heads/{branch}")["object"]["sha"]
    previous_tree = api("GET", f"/git/commits/{head}")["tree"]["sha"]
    old_tree = api("GET", f"/git/trees/{previous_tree}?recursive=1")
    if old_tree.get("truncated"):
        raise ValueError("Cannot verify a truncated GitHub source tree")
    existing = {entry["path"]: entry["sha"] for entry in old_tree["tree"] if entry["type"] == "blob"}
    entries = []
    for name, content in files.items():
        expected = git_blob(content)
        if existing.get(name) != expected:
            uploaded = api("POST", "/git/blobs", json={"content": base64.b64encode(content).decode(),
                                                        "encoding": "base64"})["sha"]
            if uploaded != expected:
                raise ValueError(f"GitHub blob digest mismatch for {name}")
            entries.append({"path": name, "mode": "100644", "type": "blob", "sha": uploaded})
    if entries:
        tree = api("POST", "/git/trees", json={"base_tree": previous_tree, "tree": entries})["sha"]
        commit = api("POST", "/git/commits", json={"message": "Implement Experiment 1.6 ranking screen and recovery",
                                                   "tree": tree, "parents": [head]})["sha"]
        api("PATCH", f"/git/refs/heads/{branch}", json={"sha": commit, "force": False})
    else:
        commit = head
    final_tree = api("GET", f"/git/trees/{api('GET', f'/git/commits/{commit}')['tree']['sha']}?recursive=1")
    if final_tree.get("truncated"):
        raise ValueError("Cannot verify a truncated GitHub source commit")
    remote = {entry["path"]: entry["sha"] for entry in final_tree["tree"] if entry["type"] == "blob"}
    if any(remote.get(name) != git_blob(content) for name, content in files.items()):
        raise ValueError("Private source commit did not match local files")
    result = {"repository": GITHUB_REPO, "private": True, "branch": branch, "commit": commit,
              "files": {name: {"bytes": len(content), "git_blob": git_blob(content)} for name, content in files.items()},
              "verified_at": datetime.now().astimezone().isoformat()}
    write_json(root / "control" / "github_source.json", result)
    return result


def git_blob_file(path: Path) -> str:
    hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def model_files(root: Path, arm: str, source_report: dict) -> dict[str, Path]:
    if arm not in ("A", "B", "C") or source_report.get("repository") != GITHUB_REPO or not source_report.get("private"):
        raise ValueError("A verified private source commit and selected arm are required")
    selected = root / arm
    if not (selected / "TRAINING_DONE.json").is_file():
        raise ValueError("Only an orderly finished selected arm may be published")
    best, latest = pointer(selected, "best"), pointer(selected, "latest")
    best_state = json.loads((best / "state.json").read_text())
    latest_state = json.loads((latest / "state.json").read_text())
    if not 0 < best_state["step"] <= latest_state["step"]:
        raise ValueError("Best/latest must contain completed optimizer updates")
    paths = {f"best/{name}": best / name for name in ("weights.safetensors", "config.json", "state.json")}
    paths.update({f"latest/{name}": latest / name for name in ("recovery.pt", "sampler.sqlite", "config.json", "state.json")})
    for rng in sorted(latest.glob("rng-*.pt")):
        paths[f"latest/{rng.name}"] = rng
    if len(list(latest.glob("rng-*.pt"))) != latest_state["world"]:
        raise ValueError("Latest recovery is missing per-rank RNG states")
    paths["vocabulary.json"] = root / "panel" / "vocabulary.json"
    for name in ("baselines.json", "evaluation_registry.json"):
        paths[f"reports/{name}"] = root / "reports" / name
    paths["reports/training_summary.json"] = selected / "reports" / "training_summary.json"
    for path in sorted((selected / "runs").rglob("events.out.tfevents.*")):
        paths[f"runs/{path.name}"] = path
    if not any(name.startswith("runs/") for name in paths):
        raise ValueError("Selected arm has no aggregate TensorBoard events")
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True, mode=0o700)
    card = (f"https://github.com/{GITHUB_REPO}/blob/{source_report['commit']}/"
            f"{GITHUB_FOLDER}/MODEL_CARD.md")
    (publication / "README.md").write_text(f"[Private model card]({card})\n")
    metadata = json.loads((root / "panel" / "meta.json").read_text())
    write_json(publication / "reconstruction.json", {
        "dataset": DATASET, "revision": REVISION, "dataset_license": "odc-by",
        "return_scale_pct": metadata["return_scale_pct"], "input_clip": metadata["input_clip"],
        "training_identities": metadata["n_tickers"], "selected_arm": arm,
        "best_step": best_state["step"], "latest_step": latest_state["step"],
        "source_commit": source_report["commit"], "vocabulary_sha256": digest(paths["vocabulary.json"])})
    paths["README.md"] = publication / "README.md"
    paths["reconstruction.json"] = publication / "reconstruction.json"
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Incomplete selected model, metadata, or aggregate logs")
    return paths


def publish_model(root: Path, arm: str) -> dict:
    source_report = json.loads((root / "control" / "github_source.json").read_text())
    paths = model_files(root, arm, source_report)
    api = HfApi(token=credential("HF_TOKEN"))
    if api.whoami().get("name") != "YL95":
        raise ValueError("HF token does not own the selected model namespace")
    dataset = api.dataset_info(DATASET, revision=REVISION)
    card = getattr(dataset, "cardData", None)
    license_name = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if dataset.id != DATASET or license_name != "odc-by":
        raise ValueError("Pinned dataset provenance or declared license changed")
    api.create_repo(repo_id=HF_REPO, repo_type="model", private=False, exist_ok=True)
    if api.model_info(HF_REPO).private:
        raise ValueError("Requested model destination is not public")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in paths.items()]
    commit = api.create_commit(repo_id=HF_REPO, repo_type="model", operations=operations,
                               commit_message=f"Publish selected {arm} best/latest and aggregate logs", num_threads=8)
    remote = {entry.rfilename: entry for entry in api.model_info(HF_REPO, revision=commit.oid,
                                                                  files_metadata=True).siblings}
    verified = {}
    for name, path in paths.items():
        entry = remote.get(name)
        if entry is None or entry.size != path.stat().st_size:
            raise ValueError(f"Public backup size verification failed for {name}")
        expected = digest(path) if entry.lfs is not None else git_blob_file(path)
        actual = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        if actual != expected:
            raise ValueError(f"Public backup digest verification failed for {name}")
        verified[name] = {"bytes": entry.size, "digest": actual}
    result = {"repository": HF_REPO, "commit": commit.oid, "selected_arm": arm,
              "verified_files": verified, "verified_at": datetime.now().astimezone().isoformat()}
    write_json(root / "reports" / "public_backup_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("source", "model"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C"), default="C")
    parser.add_argument("--plan", type=Path, default=Path.home() / "experiment_1.6_plan.md")
    arguments = parser.parse_args()
    if arguments.action == "source":
        result = github_commit(Path(__file__).resolve().parents[1], arguments.plan, arguments.root)
    else:
        result = publish_model(arguments.root, arguments.arm)
    print(json.dumps({"repository": result["repository"], "commit": result["commit"],
                      "files": len(result.get("files", result.get("verified_files", {})))}), flush=True)


if __name__ == "__main__":
    main()