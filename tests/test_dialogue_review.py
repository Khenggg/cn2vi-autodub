from dataclasses import replace

from autodub.contracts import Segment
from autodub.storage import now_ms


def seed_review(app, uploaded):
    db = app.state.db
    app.state.service.settings = replace(app.state.settings, enable_pipeline=True)
    db.execute("UPDATE episode SET status='NEEDS_REVIEW',duration_ms=3000,next_stage='TTS' WHERE id=?", (uploaded["id"],))
    segment = Segment(id="line_1", start_ms=500, end_ms=1500, zh_text="你好", dub_vi="Xin chào.",
                      words=[{"t": "你好", "s": 500, "e": 1500}], confidence={"asr": None},
                      action="NEEDS_REVIEW", needs_review=True)
    db.execute("INSERT INTO segment VALUES(?,?,?,?,?,?,?,?,?,?,?)",
               (uploaded["id"] + ":line_1", uploaded["id"], 500, 1500, segment.zh_text, "", segment.dub_vi,
                None, segment.action, "NEEDS_REVIEW", segment.model_dump_json()))
    db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
               ("review_issue", uploaded["id"], "line_1", "DURATION_REWRITE_REQUIRED", "warning", "{}", 0))
    return segment.model_dump()


def test_edit_review_and_resume_preserves_ids_and_confidence(client, app, uploaded):
    line = seed_review(app, uploaded)
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 409
    line.update(action="DUB", needs_review=False, dub_vi="Chào!", voice_id="Trúc Ly", speaker_id="narrator",
                confidence={"asr": 1.0})
    response = client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line])
    assert response.status_code == 200
    saved = response.json()[0]
    assert saved["id"] == "line_1" and saved["confidence"]["asr"] is None
    assert saved["voice_id"] == "Trúc Ly"
    assert app.state.db.one("SELECT resolved FROM issue WHERE id='review_issue'")["resolved"] == 1
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 200
    assert app.state.db.one("SELECT status FROM episode WHERE id=?", (uploaded["id"],))["status"] == "QUEUED"


def test_review_rejects_cross_episode_or_unknown_ids(client, app, uploaded):
    line = seed_review(app, uploaded)
    line.update(action="DUB", needs_review=False, id="other_episode:line_1")
    assert client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line]).status_code == 400
    assert client.get(f"/api/episodes/{uploaded['id']}/segments").json()[0]["action"] == "NEEDS_REVIEW"


def test_review_cannot_extend_words_outside_source(client, app, uploaded):
    line = seed_review(app, uploaded)
    line.update(action="DUB", needs_review=False, end_ms=4000, words=[{"t":"你好","s":500,"e":4000}])
    assert client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line]).status_code == 400


def test_active_processing_blocks_edit_and_delete(client, app, uploaded):
    line = seed_review(app, uploaded)
    app.state.db.execute("UPDATE episode SET status='TTS' WHERE id=?", (uploaded["id"],))
    assert client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line]).status_code == 409
    assert client.delete(f"/api/episodes/{uploaded['id']}").status_code == 409


def test_preview_is_authenticated_and_artifact_scoped(client, app, uploaded, tmp_path):
    path = tmp_path / "outputs" / "clip.mp4"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"preview")
    app.state.db.execute("INSERT INTO artifact VALUES(?,?,?,?,?,?,?)",
                         ("preview_id", uploaded["id"], "preview_video", "outputs/clip.mp4", "unused", 7, now_ms()))
    response = client.get(f"/api/episodes/{uploaded['id']}/preview")
    assert response.status_code == 200 and response.content == b"preview"
    client.post("/api/auth/logout")
    assert client.get(f"/api/episodes/{uploaded['id']}/preview").status_code == 401


def test_review_requires_request_protection(client, app, uploaded):
    line = seed_review(app, uploaded)
    line.update(action="KEEP", needs_review=False)
    client.headers.pop("X-Autodub-Request")
    assert client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line]).status_code == 403



def test_roi_review_requires_region_before_resume(client, app, uploaded):
    seed_review(app, uploaded)
    db = app.state.db
    db.execute("DELETE FROM segment WHERE episode_id=?", (uploaded["id"],))
    db.execute("UPDATE issue SET resolved=1 WHERE episode_id=?", (uploaded["id"],))
    db.execute("UPDATE episode SET next_stage='VISION_RENDER' WHERE id=?", (uploaded["id"],))
    db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
               ("roi_issue", uploaded["id"], None, "ROI_REQUIRED", "warning", "{}", 0))
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 409
    roi = {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.2, "scope": "series"}
    assert client.put(f"/api/episodes/{uploaded['id']}/roi", json=roi).status_code == 200
    assert db.one("SELECT resolved FROM issue WHERE id='roi_issue'")["resolved"] == 1
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 200


def test_roi_rejects_other_review_stage(client, app, uploaded):
    seed_review(app, uploaded)
    roi = {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.2}
    assert client.put(f"/api/episodes/{uploaded['id']}/roi", json=roi).status_code == 409


def test_duration_issue_requires_explicit_review_even_when_contract_is_dub(client, app, uploaded):
    line = seed_review(app, uploaded)
    line.update(action="DUB", needs_review=False)
    db = app.state.db
    db.execute("UPDATE segment SET action='DUB',contract_json=? WHERE episode_id=?",
               (Segment.model_validate(line).model_dump_json(), uploaded["id"]))
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 409
    assert client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line]).status_code == 200
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 200


def test_joint_roi_and_dialogue_review_cannot_bypass_either_gate(client, app, uploaded):
    line = seed_review(app, uploaded)
    db = app.state.db
    db.execute("UPDATE episode SET next_stage='VISION_RENDER' WHERE id=?", (uploaded["id"],))
    db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
               ("roi_issue", uploaded["id"], None, "ROI_REQUIRED", "warning", "{}", 0))
    roi = {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.2}
    assert client.put(f"/api/episodes/{uploaded['id']}/roi", json=roi).status_code == 200
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 409
    line.update(action="KEEP", needs_review=False)
    assert client.put(f"/api/episodes/{uploaded['id']}/segments", json=[line]).status_code == 200
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 200
