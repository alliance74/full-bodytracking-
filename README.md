# Full Body Tracking

Person tracking intelligence system with face and body recognition capabilities.

Software-only continuation of FaceLocking. The original projects are unchanged.
The copied ArcFace model, enrollment templates, embedding code, and five-point
alignment are reused. The new application does not import MQTT or require an ESP.

## Run

```bash
python -m src.app --camera 1
```

If the external camera cannot open, the app tries laptop camera 0. You can also
select it explicitly with `--camera 0`. The accepted horizontal mirror is retained;
`--no-mirror` disables it. Images stay upright.

The `.venv` symlink reuses the existing installed environment to avoid downloads.
On another computer, create a Python 3.11 environment, install `requirements.txt`,
copy your original ArcFace model, and run `python -m src.download_models`.
OpenCV's GUI build and Python's Tk support are needed for interactive enrollment.

## Use

1. For this clean test the active gallery starts empty. Previous registrations are
   archived under `data/backups/before-clean-test-*`; original projects are untouched.
2. For a laptop camera, keep your head, shoulders, and chest in view. Click your
   body box, then **Enroll**. Upper-body mode is the default; feet are not required.
   Enrollment automatically locks that person; clicks on other bodies cannot retarget it.
3. Enter your name. Follow each instruction: face forward, slight head left/right,
   then upper-body front, left side, right side, back, and diagonal.
4. At each pose press **Ready** (or Space). A three-second countdown runs before
   capture. Eight face samples and three samples per body view are required.
   **Ready is your confirmation of the shown pose**; the app checks visibility,
   tracking continuity, face consistency, and crop quality, not anatomical pose.
5. Face the camera again, then press **Save profile**. Face and body embeddings
   are written together in one atomic profile file. Body views persist after restart.
6. Select **Lock** for a person to follow. Green means current face confirmation;
   blue means face-anchored body continuity. Stored body-only matches are explicitly
   shown as **body gallery (unverified)**. Orange means uncertain.

Other people may remain in view during enrollment. Turn slowly and stay away
from frame edges. A brief detection gap or overlap pauses capture and preserves
completed samples. Capture resumes after stable observations. If the target gets
a new track ID, show the same person's face again: three matching observations
can recover the enrollment lock. A different or ambiguous face cannot take it
over. A mismatched face pauses face capture without deleting samples. During body
views, a weak distant/profile face does not cancel enrollment. A clear matching
face is still required before the profile can be saved.

Drafts are saved after captures to `data/db/enrollment_draft.npz`. Restarting
restores sample counts and pose progress, but never trusts an old track ID: show
the enrolling person's face again to recover the draft. **Cancel** explicitly
removes the draft. Drafts created by older versions without this feature cannot
be recovered retroactively.

Use `--body-framing full` only when head-to-feet capture is practical. Re-enroll
body views if changing framing modes so enrollment and matching use consistent
crops. Upper-body recognition may be less distinctive among identical uniforms. **Reload** discards active enrollment
and reloads profiles. **Cancel** discards unsaved enrollment.

The overlay shows current match similarity in percent and acceptance limits:

- Face: at least **62% similarity** (cosine distance at most 0.38), with a
  **6 percentage point** lead over the next enrolled name, repeated three times.
- Saved body: at least **90% similarity**, with a **10 percentage point** lead
  over the next enrolled person, repeated three times. These defaults can be
  changed with `--body-threshold` and `--body-margin` (fractions, such as 0.90).
- Scores are embedding similarities, not calibrated probabilities of identity.
  Similar uniforms can still be ambiguous. A clear conflicting face takes
  precedence over clothing; a clear unknown face is never named from clothing.

Missing tracks are remembered for eight seconds (`--max-lost`). Names carried by
continuity expire after 60 seconds without refreshed evidence (`--face-hold`). A
person fully out of view is marked **OUT OF VIEW**, with no claimed current position.
Body profiles describe the outfit at enrollment; register again after clothing
changes if body-only matching is needed. Face evidence can re-confirm identity.

## Models and limits

- [YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime):
  official 416-pixel COCO person detector, about 3.5 MB.
- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet):
  OpenCV face detection and five landmarks, about 227 KB. The 2023 model matches
  the existing OpenCV 4.11 installation.
- ArcFace: the previous project's existing 512-dimensional face model/templates.
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid): learned body appearance.
  The small MSMT17 ONNX export comes from
  [anriha/osnet_x0_25_msmt17](https://huggingface.co/anriha/osnet_x0_25_msmt17).
  Its published SHA256 is checked. This is a community export, not an official
  OSNet release. The fixed batch of 16 is handled by batching visible bodies.

People wearing similar uniforms can remain difficult to distinguish. Appearance
similarity is not a calibrated identity probability. Heavy occlusion, crossings,
clothing changes, or leaving and re-entering can require face reconfirmation.
A completely hidden or out-of-camera person cannot have a visually verified
current position. No accuracy claim is made for your employees or uniforms until
recorded scenarios from your camera have been labeled and evaluated.

Only one inference task is queued, so a slow CPU does not accumulate camera
frames. Results are drawn on the exact image processed, avoiding displaced boxes
from mixing old detections with a new camera image. No network is used at runtime.

## Offline checks and recorded video

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.download_models --check-only
.venv/bin/python -m src.app --video your_clip.mp4 --no-mirror --headless \
  --output outputs/annotated.mp4 --events outputs/events.jsonl
```

Downloads preserve `.part` files and retry from their current byte offset. Run
`.venv/bin/python -m src.download_models` again if interrupted. Model URLs, sizes,
and checksums are recorded in `models/manifest.json`.
