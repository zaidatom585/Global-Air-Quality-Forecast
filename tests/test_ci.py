"""Checks on the CI workflow and the registry overlay: least privilege, trusted actions only, pinned scanner."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
OVERLAY = ROOT / "k8s" / "overlays" / "ghcr" / "kustomization.yaml"


def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps():
    for name, job in workflow()["jobs"].items():
        for step in job["steps"]:
            yield name, step


def test_jobs_and_order():
    jobs = workflow()["jobs"]
    assert set(jobs) == {"tests", "manifests", "images"}
    assert set(jobs["images"]["needs"]) == {"tests", "manifests"}      # nothing is published if a check fails
    assert {m["name"] for m in jobs["images"]["strategy"]["matrix"]["include"]} == {"sync", "train", "api"}


def test_least_privilege():
    wf = workflow()
    assert wf["permissions"] == {"contents": "read"}
    for name, job in wf["jobs"].items():
        perms = job.get("permissions", {})
        if name == "images":
            assert perms == {"contents": "read", "packages": "write"}
        else:
            assert "packages" not in perms


def test_only_github_and_docker_actions():
    for _, step in steps():
        if "uses" in step:
            owner = step["uses"].split("/")[0]
            assert owner in {"actions", "docker"}, step["uses"]
            assert "@" in step["uses"]


def test_scanner_is_pinned_to_a_known_good_version():
    image = workflow()["env"]["TRIVY_IMAGE"]
    assert re.fullmatch(r"ghcr\.io/aquasecurity/trivy:0\.69\.3", image)


def test_publishing_never_happens_on_pull_requests():
    for _, step in steps():
        if step.get("uses", "").startswith("docker/login-action") or step.get("with", {}).get("push") is True:
            assert "pull_request" not in step.get("if", "") and step.get("if"), step


def test_registry_overlay_covers_every_image_and_container():
    o = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))
    assert {i["name"] for i in o["images"]} == {"aqcast-sync", "aqcast-train", "aqcast-api"}
    assert all(i["newName"] == i["newName"].lower() for i in o["images"])      # registry names must be lowercase
    ops = sum(len(yaml.safe_load(p["patch"])) for p in o["patches"])
    assert ops == 6                                                           # 1 API + 3 pipeline + 2 retrain containers
