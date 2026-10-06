from pathlib import Path
import numpy as np
import pytest
import yaml
import trimesh
from foundationpose_depth_gate import evaluate_depth_consistency
from foundationpose_ipc_queue import enqueue_frame, dequeue_frame, frame_files
from model_package import resolve_model, require_foundationpose_mesh


def test_fifo_preserves_frame_stamp_and_depth(tmp_path):
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.uint16) * 350
    for n in (1, 2):
        enqueue_frame(tmp_path, n, {"sequence": n, "stamp_sec": 12}, color, depth)
    meta, rgb, dep, _ = dequeue_frame(tmp_path)
    assert meta == {"sequence": 1, "stamp_sec": 12}
    np.testing.assert_array_equal(rgb, color)
    np.testing.assert_array_equal(dep, depth)
    assert len(frame_files(tmp_path)) == 1
    assert dequeue_frame(tmp_path, after_sequence=1)[0]["sequence"] == 2


def test_corrupt_frame_does_not_block_fifo(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/frame_000000000001.npz").write_bytes(b"broken")
    enqueue_frame(tmp_path, 2, {"sequence": 2}, np.zeros((2,2,3)), np.zeros((2,2)))
    assert dequeue_frame(tmp_path)[0]["sequence"] == 2
    assert not frame_files(tmp_path)


def test_depth_rejects_wrong_distance_and_missing_measurement():
    rendered = np.ones((30,30), dtype=np.float32) * .4
    assert evaluate_depth_consistency(rendered, rendered).passed
    assert not evaluate_depth_consistency(rendered + .1, rendered).passed
    assert not evaluate_depth_consistency(np.zeros_like(rendered), rendered).passed


def test_depth_shape_mismatch_is_explicit():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_depth_consistency(np.zeros((2,3)), np.zeros((3,2)))


def test_portable_mesh_and_units(tmp_path):
    trimesh.creation.box(extents=[.04,.05,.10]).export(tmp_path / "box.obj")
    meta = dict(model_id="fixture_box", mesh="box.obj", units="meters")
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(meta))
    spec = resolve_model(tmp_path)
    assert spec.model_id == "fixture_box"
    assert require_foundationpose_mesh(spec.mesh_path, convex_hull=True)["watertight"]
    meta['units'] = 'millimeters'
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="meter units"):
        resolve_model(tmp_path)


def test_inward_mesh_is_rejected(tmp_path):
    mesh = trimesh.creation.box()
    mesh.faces = mesh.faces[:, ::-1]
    mesh.export(tmp_path / "inward.obj")
    with pytest.raises(ValueError, match="invalid"):
        require_foundationpose_mesh(tmp_path / "inward.obj")
