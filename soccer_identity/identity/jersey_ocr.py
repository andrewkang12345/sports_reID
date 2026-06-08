from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from soccer_identity.utils.geometry import merge_probability_dicts, softmax
from soccer_identity.utils.schemas import RosterPlayer, Tracklet


class JerseyOCR:
    def recognize(self, crop: np.ndarray, candidate_numbers: Iterable[str]) -> tuple[dict[str, float], float]:
        raise NotImplementedError


@dataclass
class LegibilityClassifier:
    """Binary ResNet34 classifier from mkoshkina/jersey-number-pipeline that decides whether
    a player crop's jersey number is legible. Pre-filtering OCR with this cuts noise from
    blurry/occluded/back-facing frames — those used to produce noisy PARSeq reads that
    polluted the per-track jersey votes and drove "label teleport".

    Input: BGR crop. Output: probability in [0, 1] that the jersey number is legible.
    """

    model: Any
    device: str
    threshold: float = 0.5
    input_size: int = 224

    def is_legible(self, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        return self.legible_prob(crop) >= self.threshold

    def legible_prob(self, crop: np.ndarray) -> float:
        import torch as _torch
        if crop is None or crop.size == 0:
            return 0.0
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (resized - mean) / std
        tensor = _torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with _torch.no_grad():
            logit = self.model(tensor)
            prob = _torch.sigmoid(logit).item()
        return float(prob)


def build_legibility_classifier(config: dict[str, Any]) -> LegibilityClassifier | None:
    leg_cfg = config.get("legibility", {}) or {}
    if not leg_cfg.get("enabled", False):
        return None
    try:
        import torch as _torch
        from torchvision.models import resnet34
    except Exception as exc:
        print(f"[legibility] torchvision not available: {exc}; disabled.")
        return None
    weights_path = leg_cfg.get("weights", "models/mkoshkina_legibility_resnet34.pth")
    prefer_gpu = bool(leg_cfg.get("gpu", True))
    device = "cuda" if prefer_gpu and _torch.cuda.is_available() else "cpu"
    model = resnet34(weights=None)
    model.fc = _torch.nn.Linear(model.fc.in_features, 1)
    state = _torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    # mkoshkina prefix is "model_ft."; strip it.
    if any(k.startswith("model_ft.") for k in sd.keys()):
        sd = {k.replace("model_ft.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[legibility] load: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.eval().to(device)
    return LegibilityClassifier(
        model=model,
        device=device,
        threshold=float(leg_cfg.get("threshold", 0.5)),
        input_size=int(leg_cfg.get("input_size", 224)),
    )


def _kit_isolate(crop: np.ndarray, distance_threshold: float = 28.0) -> np.ndarray:
    """Isolate jersey-number pixels from the surrounding kit by replacing kit-color pixels
    with a flat kit-color background. The number, which contrasts strongly with the kit,
    stays visible while uniform texture/wrinkles/skin/grass are flattened out.

    distance_threshold is in LAB Euclidean distance (0-255). Smaller -> aggressive mask.
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return crop
    # Sample LEFT and RIGHT side strips for the kit-color reference; the digit lives
    # in the middle of the torso, while sides are nearly pure jersey material. Top/bottom
    # strips would mix in head/skin and shorts, contaminating the kit-color estimate.
    left = crop[:, : max(1, w // 5)]
    right = crop[:, -max(1, w // 5) :]
    border = np.concatenate([left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    if border.size == 0:
        return crop
    # Use a saturation-aware median to avoid letting black/shadow pixels dominate.
    kit_bgr = np.median(border, axis=0).astype(np.uint8)
    # LAB distance from kit_bgr per pixel.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    kit_lab = cv2.cvtColor(np.array([[kit_bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    diff = lab - kit_lab[None, None, :]
    dl = diff[:, :, 0] * 0.4
    da = diff[:, :, 1]
    db = diff[:, :, 2]
    dist = np.sqrt(dl * dl + da * da + db * db)
    digit_mask = dist > distance_threshold
    # Light morphological cleanup to remove speckle and join digit fragments.
    digit_mask_u8 = digit_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    digit_mask_u8 = cv2.morphologyEx(digit_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    digit_mask_u8 = cv2.morphologyEx(digit_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    digit_mask = digit_mask_u8 > 0
    output = np.broadcast_to(kit_bgr, crop.shape).copy()
    output[digit_mask] = crop[digit_mask]
    return output


@dataclass
class TemplateJerseyOCR(JerseyOCR):
    min_score: float = 0.16
    min_margin: float = 0.035
    score_temperature: float = 0.08
    debug: bool = False

    def recognize(self, crop: np.ndarray, candidate_numbers: Iterable[str]) -> tuple[dict[str, float], float]:
        candidates = [str(num) for num in candidate_numbers if str(num).strip()]
        if not candidates or crop.size == 0:
            return {}, 0.0
        region = self._jersey_region(crop)
        if region.size == 0 or region.shape[0] < 14 or region.shape[1] < 10:
            return {}, 0.0
        text_mask = self._text_mask(region)
        foreground = int(np.count_nonzero(text_mask))
        if foreground < 8:
            return {}, 0.0
        scores = [self._score_candidate(text_mask, candidate) for candidate in candidates]
        best = float(max(scores)) if scores else 0.0
        ordered = sorted(scores, reverse=True)
        margin = best - float(ordered[1]) if len(ordered) > 1 else best
        if best < self.min_score or margin < self.min_margin:
            return {}, max(0.0, min(1.0, best))
        probs_arr = softmax(scores, temperature=self.score_temperature)
        probs = {candidate: float(prob) for candidate, prob in zip(candidates, probs_arr)}
        # Drop near-zero tails to keep debug readable, then renormalize.
        probs = {key: value for key, value in probs.items() if value >= 0.02}
        total = sum(probs.values())
        if total > 0:
            probs = {key: value / total for key, value in probs.items()}
        return probs, float(min(1.0, best))

    @staticmethod
    def _jersey_region(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        y1 = int(h * 0.24)
        y2 = int(h * 0.62)
        x1 = int(w * 0.18)
        x2 = int(w * 0.82)
        return crop[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]

    @staticmethod
    def _text_mask(region: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 145)).astype(np.uint8) * 255
        if np.count_nonzero(white) >= 8:
            return cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mean = float(gray.mean())
        std = float(gray.std())
        bright = (gray > mean + max(14.0, 0.65 * std)).astype(np.uint8) * 255
        dark = (gray < mean - max(18.0, 0.85 * std)).astype(np.uint8) * 255
        # Prefer bright text, but support dark digits on light jerseys.
        mask = bright if np.count_nonzero(bright) >= np.count_nonzero(dark) else dark
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        return mask

    @staticmethod
    def _score_candidate(text_mask: np.ndarray, candidate: str) -> float:
        h, w = text_mask.shape[:2]
        mask = (text_mask > 0).astype(np.float32)
        scores: list[float] = []
        base_scale = min(w / max(16.0, 23.0 * len(candidate)), h / 34.0)
        for scale_mult in (0.72, 0.86, 1.0, 1.14, 1.28):
            font_scale = max(0.25, base_scale * scale_mult)
            thickness = max(1, int(round(font_scale * 2.0)))
            (tw, th), baseline = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if tw <= 0 or th <= 0 or tw > w * 1.15 or th > h * 1.2:
                continue
            for y_frac in (0.45, 0.52, 0.59, 0.66):
                template = np.zeros((h, w), dtype=np.uint8)
                x = int(round((w - tw) * 0.5))
                y = int(round(h * y_frac + th * 0.35))
                cv2.putText(template, candidate, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
                template = (template > 20).astype(np.float32)
                denom = float(np.sqrt(np.count_nonzero(mask) * np.count_nonzero(template)))
                if denom <= 0:
                    continue
                intersection = float(np.sum(mask * template))
                coverage = intersection / denom
                size_penalty = 1.0 - min(0.45, abs(np.count_nonzero(mask) - np.count_nonzero(template)) / max(1.0, np.count_nonzero(mask) + np.count_nonzero(template)))
                scores.append(coverage * size_penalty)
        return float(max(scores, default=0.0))


def aggregate_jersey_probs(tracklet: Tracklet) -> tuple[dict[str, float], float]:
    weighted: list[dict[str, float]] = []
    qualities: list[float] = []
    for obs in tracklet.observations:
        if not obs.jersey_probs:
            continue
        quality = max(0.01, obs.jersey_quality)
        qualities.append(quality)
        weighted.append({key: value * quality for key, value in obs.jersey_probs.items()})
    probs = merge_probability_dicts(weighted)
    if not probs:
        return {}, 0.0
    total = sum(probs.values())
    probs = {key: value / total for key, value in probs.items()}
    return probs, float(np.mean(qualities) if qualities else 0.0)


def roster_candidate_numbers(players: list[RosterPlayer]) -> list[str]:
    numbers = sorted({player.jersey_number for player in players if player.jersey_number is not None}, key=lambda item: (len(item), item))
    return [str(num) for num in numbers]


@dataclass
class EasyOCRJerseyOCR(JerseyOCR):
    reader: Any
    min_confidence: float = 0.30
    min_height_px: int = 18
    min_width_px: int = 14
    upscale_height: int = 96
    fuzzy_substring_weight: float = 0.7
    template_fallback: JerseyOCR | None = None

    def recognize(self, crop: np.ndarray, candidate_numbers: Iterable[str]) -> tuple[dict[str, float], float]:
        candidates = {str(num).strip() for num in candidate_numbers if str(num).strip()}
        if not candidates or crop.size == 0:
            return {}, 0.0
        region = self._jersey_region(crop)
        if region.size == 0 or region.shape[0] < self.min_height_px or region.shape[1] < self.min_width_px:
            if self.template_fallback is not None:
                return self.template_fallback.recognize(crop, candidate_numbers)
            return {}, 0.0
        if region.shape[0] < self.upscale_height:
            scale = self.upscale_height / region.shape[0]
            new_w = max(self.min_width_px, int(region.shape[1] * scale))
            region = cv2.resize(region, (new_w, self.upscale_height), interpolation=cv2.INTER_CUBIC)
        try:
            results = self.reader.readtext(
                region,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
            )
        except Exception:
            if self.template_fallback is not None:
                return self.template_fallback.recognize(crop, candidate_numbers)
            return {}, 0.0
        if not results:
            return {}, 0.0
        votes: dict[str, float] = {}
        best_conf = 0.0
        for _bbox, text, conf in results:
            text = str(text).strip()
            confidence = float(conf)
            if not text.isdigit():
                continue
            best_conf = max(best_conf, confidence)
            if text in candidates:
                votes[text] = max(votes.get(text, 0.0), confidence)
                continue
            for cand in candidates:
                if cand and (cand in text or text in cand):
                    votes[cand] = max(votes.get(cand, 0.0), confidence * self.fuzzy_substring_weight)
        if not votes:
            return {}, float(best_conf)
        total = sum(votes.values())
        if total <= 0:
            return {}, float(best_conf)
        probs = {key: value / total for key, value in votes.items()}
        return probs, float(best_conf)

    @staticmethod
    def _jersey_region(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h / max(1, w) <= 1.3:
            return crop  # already a torso-tight crop (pose-cropped); use as-is.
        y1 = int(h * 0.20)
        y2 = int(h * 0.62)
        x1 = int(w * 0.12)
        x2 = int(w * 0.88)
        return crop[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]


@dataclass
class ParseqSoccerNetJerseyOCR(JerseyOCR):
    """SoccerNet-fine-tuned PARSeq scene-text recognizer (from mkoshkina/jersey-number-pipeline weights)."""

    model: Any
    preprocess: Any
    device: str = "cuda"
    min_confidence: float = 0.40
    min_height_px: int = 18
    min_width_px: int = 14
    fuzzy_substring_weight: float = 0.7
    kit_isolate: bool = True
    kit_isolate_distance: float = 28.0
    strict_roster: bool = False
    multi_crop: bool = True
    multi_region: bool = True  # also try upper-torso + mid-back vertical slices
    confusable_smooth_strength: float = 0.4  # 0 disables; 0.4-0.6 boosts top-3 by ~15pp
    template_fallback: JerseyOCR | None = None

    @staticmethod
    def _confusable_smooth(probs: dict[str, float], strength: float, expand_one_digit: float = 0.6) -> dict[str, float]:
        """Redistribute mass over PARSeq digit-confusables. See eval_ocr_stack.py for the
        per-digit confusable map derived from the COCO-GT failure probe."""
        if not probs or strength <= 0:
            return probs
        DIGIT_NEIGHBORS = {
            "0": ["8", "9"], "1": ["7", "4"], "2": ["7", "3"], "3": ["8", "5", "2"],
            "4": ["1", "9", "6"], "5": ["6", "8", "3"], "6": ["5", "8", "4", "0"],
            "7": ["1", "2"], "8": ["3", "5", "9", "0", "6"], "9": ["8", "4"],
        }
        smoothed: dict[str, float] = dict(probs)
        for jersey, p in list(probs.items()):
            for i, ch in enumerate(jersey):
                for nb in DIGIT_NEIGHBORS.get(ch, []):
                    alt = jersey[:i] + nb + jersey[i+1:]
                    smoothed[alt] = smoothed.get(alt, 0.0) + p * strength
            if len(jersey) == 1:
                for lead in ("1", "2"):
                    alt = lead + jersey
                    smoothed[alt] = smoothed.get(alt, 0.0) + p * expand_one_digit
        total = sum(smoothed.values())
        if total <= 0:
            return probs
        return {k: v / total for k, v in smoothed.items()}

    def _augment_crops(self, region: np.ndarray) -> list[np.ndarray]:
        """Generate up to 5 crop variants for multi-crop ensemble OCR. Each variant gives
        PARSeq a different look at the same number; the consensus vote is more robust to
        any single-frame failure mode (motion blur, partial occlusion, glare).
        """
        h, w = region.shape[:2]
        if h < self.min_height_px or w < self.min_width_px:
            return [region]
        variants: list[np.ndarray] = [region]
        # Slightly tighter horizontal crop (10% inset) to focus on the digit core.
        tight = region[:, max(1, int(w * 0.08)) : w - max(1, int(w * 0.08))]
        if tight.size:
            variants.append(tight)
        # Wider crop expanded by 15% if there's room (catches second digit when bbox is tight).
        pad = int(w * 0.15)
        if pad > 0:
            wider = np.pad(region, ((0, 0), (pad, pad), (0, 0)), mode="edge")
            variants.append(wider)
        # Brightness-boosted variant for shaded crops.
        bright = np.clip(region.astype(np.float32) * 1.20 + 8, 0, 255).astype(np.uint8)
        variants.append(bright)
        # Mild unsharp mask to help PARSeq see digit edges.
        blurred = cv2.GaussianBlur(region, (3, 3), 0.7)
        sharp = cv2.addWeighted(region, 1.5, blurred, -0.5, 0)
        variants.append(sharp)
        return variants

    def recognize(self, crop: np.ndarray, candidate_numbers: Iterable[str]) -> tuple[dict[str, float], float]:
        candidates = {str(num).strip() for num in candidate_numbers if str(num).strip()}
        if not candidates or crop.size == 0:
            return {}, 0.0
        region = self._jersey_region(crop)
        if region.size == 0 or region.shape[0] < self.min_height_px or region.shape[1] < self.min_width_px:
            if self.template_fallback is not None:
                return self.template_fallback.recognize(crop, candidate_numbers)
            return {}, 0.0
        import torch  # local import to keep module light when backend unused

        if self.kit_isolate:
            region = _kit_isolate(region, distance_threshold=self.kit_isolate_distance)
        crop_variants = self._augment_crops(region) if self.multi_crop else [region]
        # Batch all variants in a single PARSeq forward.
        try:
            inputs = []
            for v in crop_variants:
                rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                inputs.append(self.preprocess(rgb))
            batch = torch.stack(inputs).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs_t = logits.softmax(-1)
                pred_texts, pred_confs = self.model.tokenizer.decode(probs_t)
        except Exception:
            if self.template_fallback is not None:
                return self.template_fallback.recognize(crop, candidate_numbers)
            return {}, 0.0
        if not pred_texts:
            return {}, 0.0
        # Aggregate the per-variant outputs: every digit-run from every variant gets
        # to vote. The strongest consensus wins; a single-variant misread is outvoted.
        votes: dict[str, float] = {}
        best_conf = 0.0
        for vi, text in enumerate(pred_texts):
            text = str(text)
            digit_runs = self._digit_runs(text)
            if not digit_runs:
                continue
            confs_t = pred_confs[vi]
            confs_list = confs_t.detach().cpu().tolist() if hasattr(confs_t, "detach") else list(confs_t)
            for run_text, start, end in digit_runs:
                if not run_text:
                    continue
                try:
                    run_int = int(run_text)
                except ValueError:
                    continue
                if run_int < 0 or run_int > 99:
                    continue
                char_confs = confs_list[start:end] if start < len(confs_list) else []
                if not char_confs:
                    continue
                run_conf = float(min(char_confs))
                best_conf = max(best_conf, run_conf)
                if self.strict_roster and candidates:
                    if run_text in candidates:
                        # MAX across variants — the best view of this digit wins. Sum would
                        # dilute multi-digit reads ("14" partial in 3 variants, "1" partial
                        # in 5 variants would let "1" beat "14" even when "14" is correct).
                        votes[run_text] = max(votes.get(run_text, 0.0), run_conf)
                    elif len(run_text) == 1:
                        for cand in candidates:
                            if cand and run_text in cand:
                                votes[cand] = max(votes.get(cand, 0.0), run_conf * self.fuzzy_substring_weight)
                else:
                    votes[run_text] = max(votes.get(run_text, 0.0), run_conf)
        if not votes:
            return {}, float(best_conf)
        total = sum(votes.values())
        if total <= 0:
            return {}, float(best_conf)
        probs_out = {key: value / total for key, value in votes.items()}
        # Confusable smoothing (PARSeq misreads 6→4/5/8, drops leading 1, etc. — see eval_ocr_stack)
        if self.confusable_smooth_strength > 0:
            smoothed = self._confusable_smooth(probs_out, self.confusable_smooth_strength)
            if candidates:
                # Restrict to in-roster candidates; smoothed alternatives off-roster were just
                # used to bridge confusable digits, not to introduce off-roster predictions.
                smoothed = {k: v for k, v in smoothed.items() if k in candidates}
                total_s = sum(smoothed.values())
                if total_s > 0:
                    smoothed = {k: v / total_s for k, v in smoothed.items()}
                    probs_out = smoothed
        return probs_out, float(best_conf)

    def recognize_multi_region(self, crop: np.ndarray, candidate_numbers: Iterable[str]) -> tuple[dict[str, float], float]:
        """Run recognize() on the full crop plus an upper-torso and mid-back slice; merge by max.

        This complements the existing pose-tight torso crop done in run_demo.py: when pose
        isn't available (e.g. tracker fragment) the simple vertical-slice ensemble still
        gives PARSeq three looks at the number.
        """
        if not self.multi_region:
            return self.recognize(crop, candidate_numbers)
        if crop.size == 0:
            return {}, 0.0
        h, w = crop.shape[:2]
        crops_to_try: list[np.ndarray] = [crop]
        if h > 60 and w > 30:
            upper = crop[int(h*0.18):int(h*0.55), int(w*0.10):int(w*0.90)]
            if upper.size > 0:
                crops_to_try.append(upper)
            mid = crop[int(h*0.30):int(h*0.65), int(w*0.10):int(w*0.90)]
            if mid.size > 0:
                crops_to_try.append(mid)
        merged: dict[str, float] = {}
        best_q = 0.0
        for sub in crops_to_try:
            p, q = self.recognize(sub, candidate_numbers)
            best_q = max(best_q, q)
            for k, v in p.items():
                merged[k] = max(merged.get(k, 0.0), v)
        if merged:
            total = sum(merged.values())
            if total > 0:
                merged = {k: v / total for k, v in merged.items()}
        return merged, best_q

    @staticmethod
    def _jersey_region(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h / max(1, w) <= 1.3:
            return crop  # already a torso-tight crop (pose-cropped); use as-is.
        y1 = int(h * 0.20)
        y2 = int(h * 0.62)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        return crop[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]

    @staticmethod
    def _digit_runs(text: str) -> list[tuple[str, int, int]]:
        runs: list[tuple[str, int, int]] = []
        i = 0
        while i < len(text):
            if text[i].isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                runs.append((text[i:j], i, j))
                i = j
            else:
                i += 1
        return runs


def _build_parseq_soccernet(ocr_config: dict[str, Any]) -> tuple[Any, Any, str]:
    try:
        import torch  # type: ignore
        from torchvision import transforms  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError("torch and torchvision required for parseq_soccernet backend.") from exc
    ckpt_path = str(ocr_config.get("weights", "models/parseq_soccernet.ckpt"))
    prefer_gpu = bool(ocr_config.get("gpu", True))
    device = "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
    try:
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=False, source="github", trust_repo=True)
    except Exception as exc:  # pragma: no cover - network/hub
        raise ImportError(
            "Could not load PARSeq from torch.hub. Install pytorch_lightning + timm and ensure network access."
        ) from exc
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    if any(not k.startswith("model.") for k in sd.keys()):
        sd = {f"model.{k}": v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[jersey_ocr] PARSeq SoccerNet load: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, preprocess, device


def _build_easyocr_reader(ocr_config: dict[str, Any]) -> Any:
    try:
        import easyocr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "easyocr is not installed; pip install easyocr or switch jersey_ocr.backend to 'template'."
        ) from exc
    prefer_gpu = bool(ocr_config.get("gpu", True))
    try:
        return easyocr.Reader(["en"], gpu=prefer_gpu, verbose=False)
    except Exception as gpu_err:
        if not prefer_gpu:
            raise
        # cuDNN/driver mismatch is common on shared hosts; fall back silently.
        print(f"[jersey_ocr] EasyOCR GPU init failed ({gpu_err}); falling back to CPU.")
        return easyocr.Reader(["en"], gpu=False, verbose=False)


def build_jersey_ocr(config: dict[str, Any]) -> JerseyOCR:
    ocr_config = config.get("jersey_ocr", {})
    backend = str(ocr_config.get("backend", "template")).lower()
    if backend not in {"template", "pytesseract", "easyocr", "parseq_soccernet"}:
        raise ValueError(f"Unsupported jersey OCR backend: {backend}")
    template = TemplateJerseyOCR(
        min_score=float(ocr_config.get("min_score", 0.16)),
        min_margin=float(ocr_config.get("min_margin", 0.035)),
        score_temperature=float(ocr_config.get("score_temperature", 0.08)),
    )
    if backend == "parseq_soccernet":
        model, preprocess, device = _build_parseq_soccernet(ocr_config)
        return ParseqSoccerNetJerseyOCR(
            model=model,
            preprocess=preprocess,
            device=device,
            min_confidence=float(ocr_config.get("min_confidence", 0.40)),
            min_height_px=int(ocr_config.get("min_height_px", 18)),
            min_width_px=int(ocr_config.get("min_width_px", 14)),
            fuzzy_substring_weight=float(ocr_config.get("fuzzy_substring_weight", 0.7)),
            kit_isolate=bool(ocr_config.get("kit_isolate", True)),
            kit_isolate_distance=float(ocr_config.get("kit_isolate_distance", 28.0)),
            strict_roster=bool(ocr_config.get("strict_roster", False)),
            multi_crop=bool(ocr_config.get("multi_crop", True)),
            multi_region=bool(ocr_config.get("multi_region", True)),
            confusable_smooth_strength=float(ocr_config.get("confusable_smooth_strength", 0.4)),
            template_fallback=template,
        )
    if backend == "easyocr":
        reader = _build_easyocr_reader(ocr_config)
        return EasyOCRJerseyOCR(
            reader=reader,
            min_confidence=float(ocr_config.get("min_confidence", 0.30)),
            min_height_px=int(ocr_config.get("min_height_px", 18)),
            min_width_px=int(ocr_config.get("min_width_px", 14)),
            upscale_height=int(ocr_config.get("upscale_height", 96)),
            fuzzy_substring_weight=float(ocr_config.get("fuzzy_substring_weight", 0.7)),
            template_fallback=template,
        )
    return template
