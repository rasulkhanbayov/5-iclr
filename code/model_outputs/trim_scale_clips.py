"""
Trim scale-reference clips to a short duration for VLM input. The ruler/comb
is static throughout the whole original clip (confirmed in Phase 1), so a few
seconds is enough -- showing the model a 144-second clip (02182026's scale v.mp4)
made a single inference call take 7+ minutes instead of ~1 minute, since decord
must still index/decode a much longer container even at low sample fps.
"""
import cv2
import os

TRIM_SECONDS = 3.0


def trim_clip(src_path, out_path, trim_seconds=TRIM_SECONDS):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_keep = int(fps * trim_seconds)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    n_written = 0
    for _ in range(n_keep):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n_written += 1
    cap.release()
    writer.release()
    return {"src": src_path, "out": out_path, "n_frames": n_written, "fps": fps}


if __name__ == "__main__":
    jobs = [
        ("/ephemeral/videos/02182026/scale v.mp4",
         "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_base_trim.mp4"),
        ("/ephemeral/videos/05052026/scale2.mp4",
         "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/05052026_scale2_base_trim.mp4"),
        ("/ephemeral/sgf_work/perturbed_clips/02182026_scalev_alpha0.5.mp4",
         "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_alpha0.5_trim.mp4"),
        ("/ephemeral/sgf_work/perturbed_clips/02182026_scalev_alpha2.mp4",
         "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/02182026_scalev_alpha2_trim.mp4"),
        ("/ephemeral/sgf_work/perturbed_clips/05052026_scale2_alpha0.5.mp4",
         "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/05052026_scale2_alpha0.5_trim.mp4"),
        ("/ephemeral/sgf_work/perturbed_clips/05052026_scale2_alpha2.mp4",
         "/ephemeral/sgf_work/model_outputs/trimmed_scale_clips/05052026_scale2_alpha2_trim.mp4"),
    ]
    os.makedirs("/ephemeral/sgf_work/model_outputs/trimmed_scale_clips", exist_ok=True)
    for src, out in jobs:
        result = trim_clip(src, out)
        print(result)
