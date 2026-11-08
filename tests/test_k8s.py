"""Checks on the Kubernetes manifests in k8s/: they parse, they are complete, and every container is locked down."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "k8s" / "base"
LOCAL_IMAGES = {"aqcast-sync:dev", "aqcast-train:dev", "aqcast-api:dev"}


def docs():
    out = []
    for path in sorted(K8S.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        for d in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if d:
                out.append((path.name, d))
    return out


def pod_specs():
    for name, d in docs():
        if d["kind"] == "Deployment":
            yield name, d["spec"]["template"]["spec"]
        elif d["kind"] == "CronJob":
            yield name, d["spec"]["jobTemplate"]["spec"]["template"]["spec"]


def containers():
    for name, spec in pod_specs():
        for c in spec.get("initContainers", []) + spec["containers"]:
            yield name, spec, c


def test_kustomization_lists_every_manifest():
    k = yaml.safe_load((K8S / "kustomization.yaml").read_text(encoding="utf-8"))
    files = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert set(k["resources"]) == files
    assert k["namespace"] == "aqcast"


def test_expected_objects_exist():
    kinds = {(d["kind"], d["metadata"]["name"]) for _, d in docs()}
    for want in [("Deployment", "api"), ("Service", "api"), ("CronJob", "pipeline"), ("CronJob", "retrain"),
                 ("PersistentVolumeClaim", "data"), ("PersistentVolumeClaim", "models")]:
        assert want in kinds


def test_every_container_is_locked_down():
    for file, spec, c in containers():
        where = f"{file}:{c['name']}"
        assert spec["securityContext"]["runAsNonRoot"] is True, where
        assert spec.get("automountServiceAccountToken") is False, where
        sc = c["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False, where
        assert sc["readOnlyRootFilesystem"] is True, where
        assert sc["capabilities"]["drop"] == ["ALL"], where
        assert {"cpu", "memory"} <= set(c["resources"]["limits"]), where
        assert any(m["mountPath"] == "/tmp" for m in c["volumeMounts"]), where   # writable scratch space


def test_images_are_the_ones_compose_builds():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    for file, _, c in containers():
        assert c["image"] in LOCAL_IMAGES, f"{file}:{c['name']} uses {c['image']}"
        assert f"image: {c['image']}" in compose
        assert c["imagePullPolicy"] == "Never"


def test_volumes_point_at_existing_claims():
    claims = {d["metadata"]["name"] for _, d in docs() if d["kind"] == "PersistentVolumeClaim"}
    for file, spec in pod_specs():
        for v in spec["volumes"]:
            if "persistentVolumeClaim" in v:
                assert v["persistentVolumeClaim"]["claimName"] in claims, file
        names = {v["name"] for v in spec["volumes"]}
        for c in spec.get("initContainers", []) + spec["containers"]:
            assert {m["name"] for m in c["volumeMounts"]} <= names, f"{file}:{c['name']}"


def test_api_reads_only():
    for file, spec, c in containers():
        if file == "api.yaml":
            assert all(m.get("readOnly") for m in c["volumeMounts"] if m["name"] in ("data", "models"))


def test_schedules():
    jobs = {d["metadata"]["name"]: d["spec"] for _, d in docs() if d["kind"] == "CronJob"}
    assert jobs["pipeline"]["concurrencyPolicy"] == "Forbid"
    assert jobs["pipeline"]["timeZone"] == "Etc/UTC"
    assert jobs["pipeline"].get("suspend", False) is False
    assert jobs["retrain"]["suspend"] is True          # retraining is started by hand
    # The pipeline must run after each 6-hourly issue time, once per slot.
    minute, hours = jobs["pipeline"]["schedule"].split()[:2]
    assert hours == "0,6,12,18" and 0 < int(minute) < 60


def test_api_keys_come_from_a_secret_not_the_manifests():
    text = "".join(p.read_text(encoding="utf-8") for p in K8S.glob("*.yaml"))
    assert "OPENAQ_API_KEY=" not in text and "FIRMS_MAP_KEY=" not in text
    refs = [e["secretRef"]["name"] for _, _, c in containers() for e in c.get("envFrom", []) if "secretRef" in e]
    assert refs == ["aqcast-keys"]


def test_storage_points_at_the_compose_folders():
    pvs = {d["metadata"]["name"]: d["spec"] for _, d in docs() if d["kind"] == "PersistentVolume"}
    assert pvs["aqcast-data"]["hostPath"]["path"].endswith("/smoke-forecast/data")
    assert pvs["aqcast-models"]["hostPath"]["path"].endswith("/smoke-forecast/models")
    assert all(p["persistentVolumeReclaimPolicy"] == "Retain" for p in pvs.values())


def test_no_kustomization_includes_a_folder_that_contains_it():
    """kustomize rejects that as a cycle (the ghcr overlay once used ../.. and failed in CI)."""
    for kfile in (ROOT / "k8s").rglob("kustomization.yaml"):
        here = kfile.parent.resolve()
        for entry in yaml.safe_load(kfile.read_text(encoding="utf-8")).get("resources", []):
            target = (here / entry).resolve()
            if target.is_dir():
                assert target not in here.parents, f"{kfile}: {entry} contains this kustomization"
