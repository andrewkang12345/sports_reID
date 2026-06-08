from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from soccer_identity.detection.player_detector import build_player_detector
from soccer_identity.identity.assignment import build_assigner
from soccer_identity.identity.body_reid import build_body_reid_memory, extract_body_embedding
from soccer_identity.identity.reid_extractor import load_reid_extractor, extract_appearance_batch
from soccer_identity.identity.appearance_memory import build_appearance_memory, override_jerseys_via_memory
from soccer_identity.identity.fusion import TrackletEvidence, build_identity_resolver
from soccer_identity.identity.headshot_matcher import build_headshot_matcher, extract_head_embedding
from soccer_identity.identity.jersey_ocr import build_jersey_ocr, build_legibility_classifier, roster_candidate_numbers
from soccer_identity.identity.position_prior import build_position_prior
from soccer_identity.identity.team_classifier import build_team_classifier, extract_kit_colors
from soccer_identity.tracking.tracker import build_tracker
from soccer_identity.tracking.tracklet import TrackletBuilder
from soccer_identity.utils.geometry import crop_quality, crop_xyxy
from soccer_identity.utils.schemas import Tracklet, deep_merge, load_metadata, load_yaml_config, write_json
from soccer_identity.utils.video_io import get_video_info, iter_video_frames
from soccer_identity.visualization.render import build_renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run roster-conditioned soccer player identity demo.")
    parser.add_argument("--video", required=True, help="Path to input soccer clip.")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON with teams and rosters.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_yaml_config(config_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, players = load_metadata(args.metadata)
    if not players:
        raise ValueError("Metadata rosters contain no players.")
    if metadata.get("identity_labels_available") is False:
        config = deep_merge(
            config,
            {
                "assignment": {"identity_confidence_threshold": 1.01},
                "visualization": {"likely_threshold": 1.01},
            },
        )
    # Auto-enable strict roster filtering for ground-truth-rostered clips so PARSeq
    # predictions outside the roster are discarded at voting time.
    if (metadata.get("public_source") or {}).get("roster_source") == "ground_truth_user_provided":
        config = deep_merge(config, {"jersey_ocr": {"strict_roster": True}})

    info = get_video_info(args.video)
    print(f"Loaded clip: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.2f}s")
    print(f"Loaded roster: {len(players)} players across {len(metadata.get('rosters', {}))} teams")

    detector = build_player_detector(config)
    pose_model = _load_pose_model(config)
    tracker = build_tracker(config)
    team_classifier = build_team_classifier(config, metadata)
    jersey_ocr = build_jersey_ocr(config)
    legibility = build_legibility_classifier(config)
    sr_model = _load_sr_model(config)
    if sr_model is not None:
        print(f"[sr] Real-ESRGAN x4 active for OCR input upscaling")
    if legibility is not None:
        print(f"[legibility] mkoshkina ResNet34 classifier active (threshold={legibility.threshold})")
    headshot_matcher = build_headshot_matcher(config, players, args.metadata)
    body_memory = build_body_reid_memory(config)
    # Real ReID extractor for cross-track appearance matching after tracklets are built.
    # OSNet x1.0 is light (4MB) and pedestrian-trained, plenty for 30s broadcast clips.
    appearance_cfg = config.get("appearance_memory", {})
    appearance_enabled = bool(appearance_cfg.get("enabled", True))
    appearance_stride = max(1, int(appearance_cfg.get("frame_stride", 3)))
    appearance_extractor = None
    if appearance_enabled:
        appearance_weights = str(appearance_cfg.get("reid_weights", "osnet_x1_0_msmt17.pt"))
        try:
            appearance_extractor = load_reid_extractor(weights=appearance_weights, device="cuda:0")
            print(f"[appearance] OSNet loaded ({appearance_weights}) for cross-track identity memory")
        except Exception as exc:
            print(f"[appearance] failed to load extractor: {exc} — disabling appearance memory")
            appearance_extractor = None
    position_prior = build_position_prior(config)
    resolver = build_identity_resolver(config, players, team_classifier, headshot_matcher, body_memory, position_prior)
    assigner = build_assigner(config, players, body_memory)
    renderer = build_renderer(config, players)

    # If the metadata pins which jersey numbers are visible in this clip, use that as
    # the OCR candidate set (filters PARSeq noise). Full roster is still used for name lookup.
    visible_map = metadata.get("visible_jersey_numbers") or {}
    if visible_map:
        visible_nums = sorted({str(n) for nums in visible_map.values() for n in nums}, key=lambda s: (len(s), s))
        candidate_numbers = visible_nums
        print(f"Using visible-numbers filter: {visible_nums}")
    else:
        candidate_numbers = roster_candidate_numbers(players)
    tracklet_builder = TrackletBuilder(min_observations=int(config.get("tracking", {}).get("min_tracklet_observations", 2)))
    max_seconds = config.get("video", {}).get("max_seconds", 10.0)
    jersey_stride = max(1, int(config.get("jersey_ocr", {}).get("run_every_n_frames", 2)))

    for frame_index, timestamp, frame in iter_video_frames(args.video, max_seconds=max_seconds):
        detections = detector.detect(frame, frame_index, timestamp)
        tracked = tracker.update(detections, frame_index, timestamp, frame=frame)
        if pose_model is not None and tracked and frame_index % jersey_stride == 0:
            _attach_pose_keypoints(pose_model, frame, tracked, config)

        appearance_by_tid: dict[Any, list[float] | None] = {}
        # Stride OSNet: per-frame embeddings aren't needed since the clustering only uses
        # them at top-K OCR-confident anchor frames. Running every N frames cuts ~70% of
        # OSNet calls with no quality loss.
        if (
            appearance_extractor is not None and tracked
            and frame_index % appearance_stride == 0
        ):
            bboxes = [(it.bbox.x1, it.bbox.y1, it.bbox.x2, it.bbox.y2) for it in tracked]
            embs = extract_appearance_batch(appearance_extractor, frame, bboxes)
            if embs is not None and len(embs) == len(tracked):
                for it, emb in zip(tracked, embs):
                    appearance_by_tid[it.track_id] = [float(v) for v in emb.tolist()]

        # Size-gating for OCR: tiny crops produce noisy reads that confuse downstream
        # identity inference. Catch these detections (tracker still maintains continuity)
        # but skip OCR on them — identity will be inherited from larger frames of the
        # same player via per-track GT vote / cross-track stitching.
        ocr_cfg = config.get("jersey_ocr", {})
        min_ocr_h = int(ocr_cfg.get("min_ocr_bbox_height", 0))
        min_ocr_w = int(ocr_cfg.get("min_ocr_bbox_width", 0))
        for item in tracked:
            crop = crop_xyxy(frame, item.bbox.xyxy, pad=2)
            team_color_rgb, team_quality, shorts_color_rgb, shorts_quality = extract_kit_colors(frame, item.bbox)
            bbox_h = int(item.bbox.y2 - item.bbox.y1)
            bbox_w = int(item.bbox.x2 - item.bbox.x1)
            too_small_for_ocr = (min_ocr_h > 0 and bbox_h < min_ocr_h) or (min_ocr_w > 0 and bbox_w < min_ocr_w)
            if frame_index % jersey_stride == 0:
                torso = _torso_crop_from_pose(frame, item)
                ocr_input = torso if torso is not None else crop
                if too_small_for_ocr:
                    jersey_probs, jersey_quality = {}, 0.0
                elif legibility is not None and legibility.legible_prob(ocr_input) < legibility.threshold:
                    jersey_probs, jersey_quality = {}, 0.0
                else:
                    # Super-resolve crops before OCR: 30x20 broadcast crops upsampled 4x
                    # give PARSeq a 120x80 input with learned prior detail instead of just
                    # bicubic noise. Real-ESRGAN takes ~3ms per crop.
                    full_for_ocr = _maybe_upscale(crop, sr_model)
                    # multi-region call internally tries upper-torso + mid-back slices in
                    # addition to the full crop — boosts top-3 by ~15pp on the COCO-GT eval.
                    recognize_fn = getattr(jersey_ocr, "recognize_multi_region", jersey_ocr.recognize)
                    jersey_probs, jersey_quality = recognize_fn(full_for_ocr, candidate_numbers)
                    if torso is not None:
                        torso_for_ocr = _maybe_upscale(torso, sr_model)
                        torso_probs, torso_quality = recognize_fn(torso_for_ocr, candidate_numbers)
                        merged: dict[str, float] = dict(jersey_probs)
                        for k, v in torso_probs.items():
                            merged[k] = max(merged.get(k, 0.0), v)
                        if merged:
                            s = sum(merged.values())
                            if s > 0:
                                merged = {k: v / s for k, v in merged.items()}
                        jersey_probs = merged
                        jersey_quality = max(jersey_quality, torso_quality)
            else:
                jersey_probs, jersey_quality = {}, 0.0
            head_embedding, head_quality = extract_head_embedding(crop)
            body_embedding = extract_body_embedding(crop)
            appearance_embedding = appearance_by_tid.get(item.track_id)
            tracklet_builder.add_observation(
                track_id=item.track_id,
                frame_index=frame_index,
                timestamp=timestamp,
                bbox=item.bbox,
                detection_confidence=item.confidence,
                team_color_rgb=team_color_rgb,
                team_color_quality=team_quality,
                shorts_color_rgb=shorts_color_rgb,
                shorts_color_quality=shorts_quality,
                jersey_probs=jersey_probs,
                jersey_quality=jersey_quality,
                head_embedding=head_embedding,
                head_quality=head_quality,
                body_embedding=body_embedding,
                appearance_embedding=appearance_embedding,
                crop_quality=crop_quality(crop),
                occlusion_score=0.0,
            )

    tracklets = tracklet_builder.finalize()
    print(f"Built {len(tracklets)} tracklets")
    evidence_by_track = resolver.resolve(tracklets)
    _mark_player_likelihood(tracklets, evidence_by_track, config, info.height, info.width)
    assigner.assign(tracklets)
    _refresh_evidence_after_assignment(tracklets, players, evidence_by_track, position_prior)
    # Stamp team argmax AFTER refresh (refresh rebuilds evidence dict and would otherwise erase it).
    _stamp_team_argmax(tracklets, evidence_by_track)
    # Classify non-player roles (referees, goalkeepers) from kit-color and shorts signals.
    role_cfg = config.get("role_filter", {})
    _classify_special_roles(
        tracklets, metadata, info.height, info.width,
        referee_min_hue_dist=int(role_cfg.get("referee_min_hue_dist", 35)),
        gk_min_hue_dist=int(role_cfg.get("gk_min_hue_dist", 55)),
        min_observations=int(role_cfg.get("min_observations", 5)),
        gk_require_uniform_kit=bool(role_cfg.get("gk_require_uniform_kit", True)),
    )

    tracks_summary = _tracks_summary(tracklets, players, evidence_by_track, config)
    # Mirror jersey_candidates onto tracklet.evidence so the production renderer can
    # consult them when applying the v20b "resolver+top-candidate" fallback.
    summary_by_id = {row["track_id"]: row for row in tracks_summary}
    for tr in tracklets:
        row = summary_by_id.get(tr.track_id)
        if row and row.get("jersey_candidates"):
            tr.evidence["jersey_candidates"] = row["jersey_candidates"]
        # Snapshot the original resolved_player NAME — needed for the renderer's fallback
        # when conflict resolution later nulls resolved_player_id but the resolver's
        # initial guess was still reasonable (OCR will co-validate). Restrict to players
        # whose (team, jersey) is in the visible roster so we don't resurrect identities
        # that the team-roster filter intentionally suppressed (e.g., Guido #18 ARG when
        # ARG's visible jerseys don't include #18).
        if row and row.get("resolved_player"):
            rp = row["resolved_player"]
            if isinstance(rp, dict) and rp.get("player_name") and rp.get("jersey_number") is not None:
                vis = metadata.get("visible_jersey_numbers") or {}
                team = rp.get("team_name")
                visible_for_team = {str(n) for n in (vis.get(team) or [])}
                rp_jersey = str(rp["jersey_number"])
                if not visible_for_team or rp_jersey in visible_for_team:
                    tr.evidence["resolved_player_name_snapshot"] = rp["player_name"]
                    tr.evidence["resolved_player_jersey_snapshot"] = rp_jersey
    _filter_off_roster(tracks_summary, metadata)
    team_rosters = None
    if (metadata.get("public_source") or {}).get("roster_source") == "ground_truth_user_provided":
        if metadata.get("visible_jersey_numbers"):
            team_rosters = {team: {str(n) for n in nums} for team, nums in metadata["visible_jersey_numbers"].items()}
        else:
            team_rosters = {
                team: {str(p.get("jersey_number")) for p in roster if p.get("jersey_number") is not None}
                for team, roster in (metadata.get("rosters") or {}).items()
            }
    _deduplicate_jerseys([r for r in tracks_summary if r["is_player"]], team_rosters=team_rosters)
    # Enforce identity uniqueness across simultaneously-active tracks (one Messi per frame).
    identity_threshold = float(config.get("assignment", {}).get("identity_confidence_threshold", 0.4))
    likely_threshold = float(config.get("visualization", {}).get("likely_threshold", 0.25))
    _resolve_identity_conflicts(tracks_summary, tracklets, players, metadata, identity_threshold, likely_threshold)
    # Propagate winning (team, jersey) labels to fragmented tracks that lost the dedup
    # but are clearly the same player (temporally disjoint, same team, jersey in their top-K candidates).
    _propagate_to_fragments(tracks_summary, tracklets)
    # Bridge BoT-SORT ID switches across brief occlusions: stitch tracks that end and start
    # near each other in time/space, even when the sink has no OCR signal of its own.
    n_stitched = _stitch_brief_gaps(tracks_summary, tracklets)
    if n_stitched:
        print(f"Stitched {n_stitched} fragmented track(s) across brief occlusions")
    # Appearance-memory override: cluster tracks by ReID embedding at their top-K OCR-confident
    # frames, then overwrite any track whose currently-assigned jersey conflicts with the
    # cluster's canonical (highest-scoring) jersey. This catches cases like Mbappé starting
    # as #10, getting mid-clip re-labeled #4 because PARSeq misread one frame.
    if appearance_extractor is not None:
        excluded_ids: set[str] = set()
        for tr in tracklets:
            if tr.evidence.get("role") in {"referee", "goalkeeper"}:
                excluded_ids.add(tr.track_id)
        team_of_track: dict[str, str | None] = {}
        for t in tracks_summary:
            tp = t.get("team_probs") or {}
            team_of_track[t["track_id"]] = max(tp, key=tp.get) if tp else None
        tracklets_by_id = {tr.track_id: tr for tr in tracklets}
        ap_cfg = config.get("appearance_memory", {})
        clusters, track_to_cluster = build_appearance_memory(
            tracklets,
            team_of_track,
            excluded_track_ids=excluded_ids,
            score_threshold=float(ap_cfg.get("anchor_score_threshold", 0.55)),
            k_per_jersey=int(ap_cfg.get("k_anchors_per_jersey", 3)),
            cosine_threshold=float(ap_cfg.get("cluster_cosine_threshold", 0.55)),
        )
        n_clusters = sum(1 for c in clusters if c.anchors)
        n_changed = override_jerseys_via_memory(
            tracks_summary, tracklets_by_id, clusters, track_to_cluster,
            min_override_margin=float(ap_cfg.get("min_override_margin", 0.15)),
            team_rosters=team_rosters,
        )
        if n_changed or n_clusters:
            print(f"[appearance] {n_clusters} identity cluster(s); overrode {n_changed} track(s) via memory")
    # Run global conflict pass to fixed-point: stitching+propagation chains can take several
    # iterations to fully settle (each demotion can re-route a track's primary identity).
    for _i in range(6):
        prev = sum(1 for t in tracks_summary if t.get("identity_demoted_from"))
        _resolve_identity_conflicts(tracks_summary, tracklets, players, metadata, identity_threshold, likely_threshold)
        cur = sum(1 for t in tracks_summary if t.get("identity_demoted_from"))
        if cur == prev:
            break
    # Aggressive cross-track stitching using OSNet appearance + spatial/temporal
    # continuity. Jersey numbers are visible for only a handful of frames per player,
    # so propagating identity from those frames to all fragments of the same person is
    # the key lever for tracker-GT accuracy.
    stitch_cfg = config.get("track_stitching", {})
    if stitch_cfg.get("enabled", True) and appearance_extractor is not None:
        from soccer_identity.identity.track_stitching import stitch_tracks
        player_by_id_obj = {p.player_id: p for p in players}
        stats = stitch_tracks(
            tracks_summary, tracklets, player_by_id_obj,
            metadata=metadata,
            app_threshold=float(stitch_cfg.get("app_threshold", 0.65)),
            max_gap_sec=float(stitch_cfg.get("max_gap_sec", 3.0)),
            max_overlap_sec=float(stitch_cfg.get("max_overlap_sec", 1.0)),
            max_center_dist=float(stitch_cfg.get("max_center_dist", 250.0)),
        )
        if stats["propagated_tracks"]:
            print(f"[stitch] {stats['groups']} groups, propagated identity to {stats['propagated_tracks']} tracks "
                  f"({stats['stitched_pairs']} edges among {stats['total_eligible_tracks']} eligible)")
        # One final conflict pass to clean up any newly created simultaneous identities
        _resolve_identity_conflicts(tracks_summary, tracklets, players, metadata, identity_threshold, likely_threshold)
    print(f"Player tracks: {sum(1 for t in tracks_summary if t['is_player'])} of {len(tracks_summary)}")

    result = {
        "clip_path": str(Path(args.video)),
        "home_team": metadata.get("home_team"),
        "away_team": metadata.get("away_team"),
        "identity_labels_available": metadata.get("identity_labels_available", True),
        "tracks": tracks_summary,
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "debug_tracks.json", [tracklet.to_debug_dict() for tracklet in tracklets])
    write_json(output_dir / "debug_identity_scores.json", _identity_debug(tracklets, players, evidence_by_track))
    write_json(output_dir / "debug_runtime.json", _runtime_debug(config, detector, jersey_ocr, tracker))

    jersey_by_track_id = {t["track_id"]: t.get("best_jersey_guess") for t in tracks_summary}
    renderer.render(args.video, output_dir / "visualization.mp4", tracklets, jersey_by_track_id)
    print(f"Wrote {output_dir / 'result.json'}")
    print(f"Wrote {output_dir / 'visualization.mp4'}")


def _mark_player_likelihood(
    tracklets: list[Tracklet],
    evidence_by_track: dict[str, TrackletEvidence],
    config: dict[str, Any],
    frame_height: int,
    frame_width: int,
) -> None:
    role_config = config.get("role_filter", {})
    min_player_likelihood = float(role_config.get("min_player_likelihood", 0.50))
    min_duration = float(role_config.get("min_track_duration_sec", 0.20))
    edge_margin = float(role_config.get("edge_margin_pixels", 4.0))
    max_edge_touch_ratio = float(role_config.get("max_edge_touch_ratio", 0.30))
    lower_sideline_y_ratio = float(role_config.get("lower_sideline_y_ratio", 0.965))
    upper_sideline_y_ratio = float(role_config.get("upper_sideline_y_ratio", 0.04))
    side_sideline_x_ratio = float(role_config.get("side_sideline_x_ratio", 0.015))
    sideline_penalty = float(role_config.get("sideline_penalty", 0.15))
    for tracklet in tracklets:
        evidence = evidence_by_track.get(tracklet.track_id)
        likelihood = evidence.player_likelihood if evidence is not None else 0.5
        duration_score = min(1.0, max(0.0, tracklet.duration / max(min_duration, 1e-6)))
        tracklet.player_likelihood = float(max(0.0, min(1.0, 0.85 * likelihood + 0.15 * duration_score)))
        if tracklet.observations and frame_height > 0 and frame_width > 0:
            n = len(tracklet.observations)
            bottom_touch = sum(1 for obs in tracklet.observations if obs.bbox.y2 >= frame_height - edge_margin)
            top_touch = sum(1 for obs in tracklet.observations if obs.bbox.y1 <= edge_margin)
            left_touch = sum(1 for obs in tracklet.observations if obs.bbox.x1 <= edge_margin)
            right_touch = sum(1 for obs in tracklet.observations if obs.bbox.x2 >= frame_width - edge_margin)
            bottoms = sorted(obs.bbox.y2 for obs in tracklet.observations)
            tops = sorted(obs.bbox.y1 for obs in tracklet.observations)
            lefts = sorted(obs.bbox.x1 for obs in tracklet.observations)
            rights = sorted(obs.bbox.x2 for obs in tracklet.observations)
            median_bottom = bottoms[n // 2]
            median_top = tops[n // 2]
            median_left = lefts[n // 2]
            median_right = rights[n // 2]
            bottom_ratio = bottom_touch / n
            top_ratio = top_touch / n
            left_ratio = left_touch / n
            right_ratio = right_touch / n
            sideline_flag = (
                bottom_ratio > max_edge_touch_ratio
                or top_ratio > max_edge_touch_ratio
                or left_ratio > max_edge_touch_ratio
                or right_ratio > max_edge_touch_ratio
                or median_bottom >= frame_height * lower_sideline_y_ratio
                or median_top <= frame_height * upper_sideline_y_ratio
                or median_left <= frame_width * side_sideline_x_ratio
                or median_right >= frame_width * (1.0 - side_sideline_x_ratio)
            )
            if sideline_flag:
                tracklet.player_likelihood *= sideline_penalty
                tracklet.evidence["sideline_staff_penalty"] = 1.0
            tracklet.evidence["edge_touch_bottom"] = float(bottom_ratio)
            tracklet.evidence["edge_touch_top"] = float(top_ratio)
            tracklet.evidence["edge_touch_left"] = float(left_ratio)
            tracklet.evidence["edge_touch_right"] = float(right_ratio)
        tracklet.is_player = tracklet.player_likelihood >= min_player_likelihood and tracklet.duration >= min_duration
        tracklet.evidence["player_likelihood"] = tracklet.player_likelihood


def _classify_special_roles(
    tracklets: list[Tracklet],
    metadata: dict[str, Any],
    frame_height: int,
    frame_width: int,
    referee_min_hue_dist: int = 35,
    gk_min_hue_dist: int = 55,
    min_observations: int = 5,
    gk_require_uniform_kit: bool = True,
) -> None:
    """Tag tracklets as 'referee' or 'goalkeeper' by HSV hue-distance from both team shirts.

    HSV hue is robust to brightness washout: a grayed-out navy still has navy hue, so a
    real French player's extracted color hashes to ~the same hue as the France reference.
    A red referee shirt has hue ~0/180 — far (>=35deg) from both light-blue and navy.

    Hue is only meaningful when BOTH observed and reference have decent saturation; if the
    observed color is washed out (sat<30), we skip the role test for that track.

    - referee: hue far from both team shirts AND shorts dark (black)
    - goalkeeper: hue far from both team shirts (kit usually pink/yellow/teal)
    """
    import cv2 as _cv2
    from soccer_identity.utils.geometry import parse_hex_color
    import numpy as np

    team_colors = metadata.get("team_colors") or {}
    team_shirt_refs: dict[str, np.ndarray] = {}
    for team, refs in team_colors.items():
        if isinstance(refs, dict) and "shirt" in refs:
            try:
                team_shirt_refs[team] = parse_hex_color(str(refs["shirt"]))
            except Exception:
                continue
    if not team_shirt_refs:
        return

    def to_hsv(rgb: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return _cv2.cvtColor(arr, _cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]

    team_hsv = {team: to_hsv(rgb) for team, rgb in team_shirt_refs.items()}

    def hue_dist(h1: float, h2: float) -> float:
        diff = abs(h1 - h2)
        return min(diff, 180.0 - diff)

    def min_hue_dist_to_teams(shirt_rgb: np.ndarray) -> tuple[float, float]:
        """Returns (min_hue_dist, observed_saturation). High sat + large hue dist = different kit."""
        hsv = to_hsv(shirt_rgb)
        obs_h, obs_s, obs_v = hsv[0], hsv[1], hsv[2]
        if obs_s < 30:
            return 0.0, float(obs_s)
        dists = [hue_dist(obs_h, team_hsv[team][0]) for team in team_hsv]
        return float(min(dists)), float(obs_s)

    n_referees = 0
    n_gk = 0
    for tracklet in tracklets:
        if not tracklet.observations:
            continue
        shirts: list[np.ndarray] = []
        shorts: list[np.ndarray] = []
        for obs in tracklet.observations:
            if obs.team_color_rgb is not None:
                shirts.append(np.asarray(obs.team_color_rgb, dtype=np.float32))
            if obs.shorts_color_rgb is not None:
                shorts.append(np.asarray(obs.shorts_color_rgb, dtype=np.float32))
        if not shirts:
            continue
        # Require a sustained track before assigning a role — single-frame false positives
        # (partial occlusion, bad crop) caused v18's 7-goalkeeper-track inflation.
        if len(tracklet.observations) < min_observations:
            continue
        shirt_med = np.median(np.stack(shirts), axis=0)
        shorts_med = np.median(np.stack(shorts), axis=0) if shorts else None
        min_hue_d, sat = min_hue_dist_to_teams(shirt_med)
        shorts_brightness = float(np.mean(shorts_med)) if shorts_med is not None else 255.0
        shorts_dark = shorts_brightness < 100
        # Goalkeepers typically wear a uniform kit (shirt + shorts same hue/color);
        # referees wear a distinct shirt + DARK shorts. Use shorts-shirt hue similarity
        # to disambiguate when shirt hue is far from both team kits.
        shorts_hue_close = False
        if shorts_med is not None:
            sh_hsv = to_hsv(shorts_med)
            sk_hsv = to_hsv(shirt_med)
            if sh_hsv[1] >= 30:  # shorts have hue (not black/gray)
                shorts_hue_close = hue_dist(sh_hsv[0], sk_hsv[0]) <= 20
        role: str | None = None
        if sat >= 30:
            if min_hue_d >= referee_min_hue_dist and shorts_dark:
                role = "referee"
            elif min_hue_d >= gk_min_hue_dist and (not gk_require_uniform_kit or (shorts_hue_close and not shorts_dark)):
                # When gk_require_uniform_kit is True (default), GK must have shirt+shorts
                # of similar hue (the uniform-kit signature) — this prevents tagging a
                # red-shirt referee as GK. When False, only hue distance to teams matters
                # (v18 behavior — looser, fewer GK false negatives on the real Emiliano).
                role = "goalkeeper"
        if role is not None:
            tracklet.evidence["role"] = role
            tracklet.evidence["role_hue_dist"] = round(float(min_hue_d), 1)
            tracklet.evidence["role_shirt_sat"] = round(float(sat), 1)
            if role == "referee":
                tracklet.player_likelihood = min(tracklet.player_likelihood, 0.15)
                tracklet.is_player = False
                n_referees += 1
            else:
                n_gk += 1
            # Goalkeepers and referees must NOT compete for outfield player-jersey slots
            # in the cross-track dedup — otherwise a misclassified track can steal a real
            # player's jersey (e.g. the second referee, classified as GK due to non-dark
            # shorts, stole Rabiot's #14 in v17). Mark them so dedup skips them.
            tracklet.evidence["excluded_from_jersey_dedup"] = True
    if n_referees or n_gk:
        print(f"Role classifier: {n_referees} referee track(s), {n_gk} goalkeeper track(s)")


def _stamp_team_argmax(tracklets: list[Tracklet], evidence_by_track: dict[str, TrackletEvidence]) -> None:
    """Persist the per-track most-likely team on the tracklet evidence dict so the renderer
    can break (team, jersey) ties (e.g., #10 -> Messi for Argentina, Mbappé for France)."""
    for tracklet in tracklets:
        evidence = evidence_by_track.get(tracklet.track_id)
        if evidence is None or not evidence.team_probs:
            continue
        team = max(evidence.team_probs, key=evidence.team_probs.get)
        tracklet.evidence["team_argmax"] = team


def _refresh_evidence_after_assignment(
    tracklets: list[Tracklet],
    players: list[Any],
    evidence_by_track: dict[str, TrackletEvidence],
    position_prior: Any,
) -> None:
    player_by_id = {player.player_id: player for player in players}
    for tracklet in tracklets:
        player = player_by_id.get(tracklet.resolved_player_id or "")
        evidence = evidence_by_track.get(tracklet.track_id)
        if player is None or evidence is None:
            continue
        jersey_conf = evidence.jersey_probs.get(player.jersey_number or "", 0.0) if evidence.jersey_probs else 0.0
        tracklet.evidence = {
            "team_confidence": float(evidence.team_probs.get(player.team_name, 0.0)),
            "jersey_confidence": float(jersey_conf),
            "headshot_confidence": float(evidence.headshot_sims.get(player.player_id, 0.0)),
            "body_reid_confidence": float(evidence.body_reid_confidence),
            "tracking_confidence": float(evidence.tracking_confidence),
            "position_prior": float(position_prior.score(tracklet, player)),
            "player_likelihood": float(tracklet.player_likelihood),
        }


def _identity_debug(
    tracklets: list[Tracklet],
    players: list[Any],
    evidence_by_track: dict[str, TrackletEvidence],
) -> list[dict[str, Any]]:
    player_by_id = {player.player_id: player for player in players}
    rows = []
    for tracklet in tracklets:
        evidence = evidence_by_track.get(tracklet.track_id)
        rows.append(
            {
                "track_id": tracklet.track_id,
                "start_time": tracklet.start_time,
                "end_time": tracklet.end_time,
                "resolved_player_id": tracklet.resolved_player_id,
                "resolved_player": player_by_id[tracklet.resolved_player_id].to_dict()
                if tracklet.resolved_player_id in player_by_id
                else None,
                "resolved_confidence": tracklet.resolved_confidence,
                "posterior": tracklet.identity_posterior,
                "raw_scores": tracklet.identity_scores,
                "evidence": {
                    "team_probs": evidence.team_probs if evidence else {},
                    "jersey_probs": evidence.jersey_probs if evidence else {},
                    "jersey_quality": evidence.jersey_quality if evidence else 0.0,
                    "headshot_sims": evidence.headshot_sims if evidence else {},
                    "tracking_confidence": evidence.tracking_confidence if evidence else 0.0,
                    "body_reid_confidence": evidence.body_reid_confidence if evidence else 0.0,
                    "player_likelihood": evidence.player_likelihood if evidence else tracklet.player_likelihood,
                },
                "is_player": tracklet.is_player,
                "player_likelihood": tracklet.player_likelihood,
            }
        )
    return rows


def _runtime_debug(config: dict[str, Any], detector: Any, jersey_ocr: Any, tracker: Any) -> dict[str, Any]:
    return {
        "detector_class": detector.__class__.__name__,
        "jersey_ocr_class": jersey_ocr.__class__.__name__,
        "tracker_class": tracker.__class__.__name__,
        "configured_detector_backend": config.get("detector", {}).get("backend"),
        "configured_detector_weights": config.get("detector", {}).get("weights"),
        "configured_jersey_backend": config.get("jersey_ocr", {}).get("backend"),
        "configured_tracker_backend": config.get("tracker", {}).get("backend"),
        "notes": [
            "YOLO COCO person weights are used when detector.backend is ultralytics.",
            "Tracking association uses BoT-SORT (Ultralytics botsort.yaml, sparseOptFlow GMC) for broadcast camera motion.",
            "Ball detection and near-ball event extraction have been removed; the demo emits per-track identity summaries.",
            "Jersey number recognition uses EasyOCR (digits-only) by default; SoccerNet jersey model can be plugged in via jersey_ocr.backend.",
            "Rendered tracks are filtered by metadata kit colors and a four-edge sideline guard.",
            "Low-confidence identities are labeled explicitly as Low conf ID in the visualization.",
        ],
    }


def _tracks_summary(
    tracklets: list[Tracklet],
    players: list[Any],
    evidence_by_track: dict[str, TrackletEvidence],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    player_by_id = {player.player_id: player for player in players}
    viz_cfg = config.get("visualization", {})
    min_jersey_display = float(viz_cfg.get("min_jersey_display_confidence", 0.30))
    peak_threshold = float(viz_cfg.get("peak_lock_threshold", 0.80))
    peak_min_count = int(viz_cfg.get("peak_lock_min_count", 2))
    longer_lock_min_count = int(viz_cfg.get("longer_lock_min_count", 2))
    # IDF calibration over the whole clip: a jersey that PARSeq peaks for in many tracks
    # is more likely a model bias than a real read. Boost rare jerseys, discount common ones.
    use_idf = bool(viz_cfg.get("idf_calibration", True))
    idf_weights = _compute_idf_weights(tracklets, peak_threshold) if use_idf else None
    if idf_weights:
        top_bias = sorted(idf_weights.items(), key=lambda kv: kv[1])[:5]
        top_rare = sorted(idf_weights.items(), key=lambda kv: -kv[1])[:5]
        print(f"[IDF] most-biased jerseys (low weight): {[(j, round(w, 2)) for j, w in top_bias]}")
        print(f"[IDF] rare jerseys (high weight): {[(j, round(w, 2)) for j, w in top_rare]}")
    rows: list[dict[str, Any]] = []
    for tracklet in tracklets:
        evidence = evidence_by_track.get(tracklet.track_id)
        player = player_by_id.get(tracklet.resolved_player_id or "")
        jersey_guess, jersey_score = _best_jersey_for_summary(tracklet, min_jersey_display, peak_threshold, peak_min_count, longer_lock_min_count, idf_weights=idf_weights)
        ranked_cands = _ranked_jersey_candidates(tracklet, min_jersey_display, peak_threshold, peak_min_count, longer_lock_min_count, idf_weights=idf_weights)
        rows.append(
            {
                "track_id": tracklet.track_id,
                "start_time": round(float(tracklet.start_time), 3),
                "end_time": round(float(tracklet.end_time), 3),
                "duration": round(float(tracklet.duration), 3),
                "is_player": bool(tracklet.is_player),
                "player_likelihood": round(float(tracklet.player_likelihood), 4),
                "resolved_player_id": tracklet.resolved_player_id,
                "resolved_player": player.to_dict() if player is not None else None,
                "resolved_confidence": round(float(tracklet.resolved_confidence), 4),
                "best_jersey_guess": jersey_guess,
                "jersey_vote_score": round(float(jersey_score), 4),
                "jersey_candidates": [(j, round(float(s), 4)) for j, s in ranked_cands[:5]],
                "jersey_peak_counts": _peak_counts(tracklet, peak_threshold),
                "role": tracklet.evidence.get("role") if isinstance(tracklet.evidence, dict) else None,
                "team_probs": {
                    team: round(float(prob), 4)
                    for team, prob in (evidence.team_probs.items() if evidence else [])
                },
            }
        )
    return rows


def _peak_counts(tracklet: Tracklet, peak_threshold: float) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in tracklet.observations:
        for jersey, prob in obs.jersey_probs.items():
            contribution = prob * max(0.01, obs.jersey_quality)
            if contribution >= peak_threshold:
                counts[jersey] = counts.get(jersey, 0) + 1
    return counts


def _stitch_brief_gaps(
    tracks: list[dict[str, Any]],
    tracklets: list[Tracklet],
    max_gap_sec: float = 2.0,
    max_center_dist: float = 200.0,
) -> int:
    """Bridge BoT-SORT ID switches across brief occlusions.

    For each labeled (jersey or resolved-player) track A, find a later unlabeled track B
    such that A ends and B starts within `max_gap_sec` and their bbox centers are within
    `max_center_dist` pixels. Same team required. Propagate A's jersey to B. Returns the
    number of stitches applied.
    """
    from soccer_identity.utils.geometry import bbox_center, point_distance

    by_tid = {tr.track_id: tr for tr in tracklets}

    def team_of(t: dict[str, Any]) -> str | None:
        tp = t.get("team_probs") or {}
        return max(tp, key=tp.get) if tp else None

    def has_label(t: dict[str, Any]) -> bool:
        if t.get("best_jersey_guess"):
            return True
        return False

    def last_bbox(tid: str):
        tr = by_tid.get(tid)
        if tr is None or not tr.observations:
            return None
        return max(tr.observations, key=lambda o: o.frame_index).bbox.xyxy

    def first_bbox(tid: str):
        tr = by_tid.get(tid)
        if tr is None or not tr.observations:
            return None
        return min(tr.observations, key=lambda o: o.frame_index).bbox.xyxy

    # Build candidate sources (labeled) and sinks (unlabeled). Iterate until no more stitches.
    stitches = 0
    for _round in range(5):
        sources = sorted([t for t in tracks if t.get("is_player") and has_label(t)],
                         key=lambda t: -float(t.get("jersey_vote_score", 0.0)))
        sinks = [t for t in tracks if t.get("is_player") and not has_label(t)]
        if not sources or not sinks:
            break
        new_stitches = 0
        for source in sources:
            stid = source["track_id"]
            s_tr = by_tid.get(stid)
            if s_tr is None or not s_tr.observations:
                continue
            s_last_bbox = last_bbox(stid)
            s_end = s_tr.end_time
            s_team = team_of(source)
            best_sink = None
            best_dist = float("inf")
            for sink in sinks:
                if sink.get("best_jersey_guess"):
                    continue  # picked up by an earlier source this round
                if sink.get("role") in {"referee", "goalkeeper"}:
                    continue  # never propagate a player label to a referee/GK track
                if not sink.get("is_player", True):
                    continue
                ktid = sink["track_id"]
                k_tr = by_tid.get(ktid)
                if k_tr is None or not k_tr.observations:
                    continue
                gap = k_tr.start_time - s_end
                if gap < 0 or gap > max_gap_sec:
                    continue
                k_team = team_of(sink)
                if s_team and k_team and s_team != k_team:
                    continue
                k_first_bbox = first_bbox(ktid)
                dist = point_distance(bbox_center(s_last_bbox), bbox_center(k_first_bbox))
                if dist > max_center_dist:
                    continue
                # Bonus check: sink's team-fit shouldn't be far worse than source's. A track
                # whose kit color doesn't match either team well (likely a referee misclassified
                # as a player) shouldn't inherit a player identity.
                if isinstance(k_tr.evidence, dict) and k_tr.evidence.get("role") in {"referee", "goalkeeper"}:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_sink = sink
            if best_sink is not None:
                best_sink["best_jersey_guess"] = source["best_jersey_guess"]
                best_sink["jersey_stitched_from"] = stid
                # Mirror onto the live Tracklet's evidence so the renderer's jersey lookup
                # disambiguates to the correct team-side of a shared number.
                k_tr_obj = by_tid.get(best_sink["track_id"])
                if k_tr_obj is not None and s_team:
                    k_tr_obj.evidence["team_argmax"] = s_team
                new_stitches += 1
        stitches += new_stitches
        if new_stitches == 0:
            break
    return stitches


def _resolve_identity_conflicts(
    tracks: list[dict[str, Any]],
    tracklets: list[Tracklet],
    players: list[Any],
    metadata: dict[str, Any],
    identity_threshold: float,
    likely_threshold: float = 0.25,
) -> None:
    """Final dedup pass: enforce that no two simultaneously-active tracks display the same
    player identity. The WindowHungarianAssigner does per-window dedup but its cross-window
    vote step can still assign the same player to multiple tracks; this catches the rest.

    For each candidate identity (resolved_player_id OR jersey-derived player_id), find
    pairs of tracks with overlapping observations and demote all but the highest-scoring
    one. Demoted tracks: their jersey can stay (it's just a number), but the player-name
    binding is cleared so the renderer falls back to "Low conf ID #X".
    """
    visible_map = metadata.get("visible_jersey_numbers") or {}
    visible_per_team = {team: {str(n) for n in nums} for team, nums in visible_map.items()}
    player_by_id = {p.player_id: p for p in players}
    by_tid = {tr.track_id: tr for tr in tracklets}

    def primary_identity(t: dict[str, Any]) -> str | None:
        # Mirror the renderer's _label logic: resolver-assigned player only counts when OCR
        # confirms the jersey number. Otherwise we fall to the jersey-derived player so the
        # conflict pass groups tracks by the identity that the renderer will actually display.
        rp_id = t.get("resolved_player_id")
        jersey = t.get("best_jersey_guess")
        if rp_id and t.get("resolved_confidence", 0.0) >= likely_threshold:
            rp = player_by_id.get(rp_id)
            if rp is not None and jersey is not None and str(rp.jersey_number) == str(jersey):
                return rp_id
            # OCR didn't confirm the resolver's assignment — fall through to jersey path.
        if not jersey:
            return None
        tp = t.get("team_probs") or {}
        team = max(tp, key=tp.get) if tp else None
        if not team:
            return None
        for p in player_by_id.values():
            if p.team_name == team and str(p.jersey_number) == str(jersey):
                return p.player_id
        return None

    def score(t: dict[str, Any]) -> float:
        return float(t.get("resolved_confidence", 0.0)) + float(t.get("jersey_vote_score", 0.0)) * 0.05 + float(t.get("duration", 0.0)) * 0.01

    # Group tracks by primary identity.
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for t in tracks:
        if not t.get("is_player"):
            continue
        pid = primary_identity(t)
        if pid is None:
            continue
        by_identity.setdefault(pid, []).append(t)

    for pid, claimants in by_identity.items():
        if len(claimants) <= 1:
            continue
        # Build observation frame sets per claimant.
        frames_per: dict[str, set[int]] = {}
        for t in claimants:
            tr = by_tid.get(t["track_id"])
            if tr is None:
                continue
            frames_per[t["track_id"]] = {obs.frame_index for obs in tr.observations}
        # Sort by score descending; greedily keep the strongest, demote any that overlap.
        # Only demote when the overlap dominates the candidate's own lifetime (>0.6) —
        # otherwise the candidate has substantial UNIQUE frames where it's the only
        # claimant and should keep its identity for those frames. The renderer handles
        # the simultaneous-label case via per-frame conflict resolution below.
        claimants.sort(key=score, reverse=True)
        kept_frames: set[int] = set()
        kept_tid: str | None = None
        for t in claimants:
            tid = t["track_id"]
            my_frames = frames_per.get(tid, set())
            if kept_tid is None:
                kept_tid = tid
                kept_frames = set(my_frames)
                continue
            shared = my_frames & kept_frames
            overlap_ratio = len(shared) / max(1, len(my_frames))
            if overlap_ratio > 0.6:
                t["resolved_player_id"] = None
                t["resolved_confidence"] = 0.0
                t["identity_demoted_from"] = pid
                # Mirror onto the live Tracklet object so the renderer sees the demotion.
                tracklet_obj = by_tid.get(tid)
                if tracklet_obj is not None:
                    tracklet_obj.resolved_player_id = None
                    tracklet_obj.resolved_confidence = 0.0
                # Clear the jersey too if it would re-derive the same identity. Otherwise
                # keep it (a different team's player with same number, or just a number).
                jersey = t.get("best_jersey_guess")
                if jersey:
                    tp = t.get("team_probs") or {}
                    team = max(tp, key=tp.get) if tp else None
                    demoted_player = player_by_id.get(pid)
                    if demoted_player and team == demoted_player.team_name and str(demoted_player.jersey_number) == str(jersey):
                        t["best_jersey_guess"] = None
                        t["jersey_demoted_with_identity"] = True
            else:
                # No frame overlap with the kept track -> safe to keep as a separate fragment.
                kept_frames |= my_frames


def _propagate_to_fragments(
    tracks: list[dict[str, Any]],
    tracklets: list[Tracklet],
    max_overlap_sec: float = 0.5,
    min_candidate_score_ratio: float = 0.20,
) -> None:
    """Once dedup has selected winners per (team, jersey), find OTHER tracks that look
    like the same player and propagate the label to them.

    A track T can inherit (team, jersey) from winner W when:
      - T is not the winner, T has no assigned jersey of its own
      - T's argmax team equals W's argmax team
      - W's jersey is in T's jersey_candidates with score >= min_ratio * W's score
      - T and W don't significantly overlap in time (occluded fragments don't co-exist)
    Multiple losers may inherit from the same winner; visualization will show them all
    with the same player name.
    """
    by_tid = {tr.track_id: tr for tr in tracklets}
    summary_by_tid = {t["track_id"]: t for t in tracks}

    def team_of(t: dict[str, Any]) -> str | None:
        tp = t.get("team_probs") or {}
        return max(tp, key=tp.get) if tp else None

    def time_window(tid: str) -> tuple[float, float] | None:
        tr = by_tid.get(tid)
        if tr is None:
            return None
        return (float(tr.start_time), float(tr.end_time))

    # Build list of winners (track with a final jersey).
    winners = [t for t in tracks if t.get("is_player") and t.get("best_jersey_guess")]
    winners.sort(key=lambda t: -float(t.get("jersey_vote_score", 0.0)))
    for w in winners:
        w_tid = w["track_id"]
        w_team = team_of(w)
        w_jersey = w["best_jersey_guess"]
        w_score = float(w.get("jersey_vote_score", 0.0))
        w_window = time_window(w_tid)
        if w_window is None:
            continue
        for other in tracks:
            tid = other["track_id"]
            if tid == w_tid:
                continue
            if not other.get("is_player"):
                continue
            if other.get("best_jersey_guess"):
                continue  # already labeled with its own jersey
            if team_of(other) != w_team:
                continue
            # Check temporal disjointness.
            o_window = time_window(tid)
            if o_window is None:
                continue
            overlap = max(0.0, min(w_window[1], o_window[1]) - max(w_window[0], o_window[0]))
            if overlap > max_overlap_sec:
                continue
            # Check that the winner's jersey was a plausible candidate for this track.
            cands = other.get("jersey_candidates") or []
            matched = next(((j, s) for j, s in cands if j == w_jersey), None)
            if matched is None:
                continue
            _j, c_score = matched
            if w_score > 0 and c_score < w_score * min_candidate_score_ratio:
                continue
            other["best_jersey_guess"] = w_jersey
            other["jersey_inherited_from"] = w_tid


def _filter_off_roster(tracks: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    src = (metadata.get("public_source") or {}).get("roster_source")
    if src != "ground_truth_user_provided":
        return
    valid: set[str] = set()
    for team, roster in (metadata.get("rosters") or {}).items():
        for p in roster:
            num = p.get("jersey_number")
            if num is not None:
                valid.add(str(num))
    if not valid:
        return
    for t in tracks:
        guess = t.get("best_jersey_guess")
        if guess and guess not in valid:
            t["best_jersey_guess"] = None
            t["jersey_off_roster"] = True


def _load_sr_model(config: dict[str, Any]) -> Any:
    """Load Real-ESRGAN x4 super-resolution model for upscaling torso crops before OCR.
    Player crops in broadcast soccer are ~30x20 pixels — too small for fine digit detail.
    SR (rather than bicubic) hallucinates plausible high-res detail from a learned prior."""
    sr_cfg = config.get("super_resolution", {}) or {}
    if not sr_cfg.get("enabled", False):
        return None
    try:
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    except Exception as exc:
        print(f"[sr] realesrgan not available: {exc}; disabled.")
        return None
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
    weights = sr_cfg.get("weights") or "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    upsampler = RealESRGANer(
        scale=4, model_path=weights, model=model,
        tile=0, tile_pad=10, pre_pad=0, half=False, gpu_id=0,
    )
    return upsampler


def _maybe_upscale(crop: Any, upsampler: Any) -> Any:
    """Super-resolve a crop with Real-ESRGAN if the upsampler is available and the crop
    is small enough to benefit (very large crops would just consume time)."""
    if upsampler is None or crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h > 96 or w > 96:
        return crop  # already large enough
    try:
        out, _ = upsampler.enhance(crop, outscale=4)
        return out
    except Exception:
        return crop


def _load_pose_model(config: dict[str, Any]) -> Any:
    pose_cfg = config.get("pose", {}) or {}
    if not pose_cfg.get("enabled", False):
        return None
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        print("[pose] ultralytics not installed; pose disabled.")
        return None
    weights = str(pose_cfg.get("weights", "yolo11m-pose.pt"))
    try:
        return YOLO(weights)
    except Exception as exc:
        print(f"[pose] could not load {weights} ({exc}); pose disabled.")
        return None


def _attach_pose_keypoints(pose_model: Any, frame: Any, tracked: list, config: dict[str, Any]) -> None:
    """Batch-run YOLO11-pose on each tracked detection's crop and attach keypoints in frame coords."""
    pose_cfg = config.get("pose", {}) or {}
    target_h = int(pose_cfg.get("crop_height", 320))
    conf_floor = float(pose_cfg.get("conf", 0.05))
    device = pose_cfg.get("device", config.get("detector", {}).get("device", "cuda"))
    crops = []
    keep = []
    offsets = []
    fh, fw = frame.shape[:2]
    for item in tracked:
        x1, y1, x2, y2 = [int(v) for v in item.bbox.xyxy]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(fw, x2); y2 = min(fh, y2)
        c = frame[y1:y2, x1:x2]
        if c.shape[0] < 8 or c.shape[1] < 8:
            continue
        scale = target_h / max(1, c.shape[0])
        new_w = max(8, int(c.shape[1] * scale))
        cr = __import__("cv2").resize(c, (new_w, target_h))
        crops.append(cr)
        keep.append(item)
        offsets.append((x1, y1, c.shape[1] / new_w, c.shape[0] / target_h))
    if not crops:
        return
    results = pose_model.predict(crops, device=device, verbose=False, imgsz=target_h, conf=conf_floor, batch=len(crops))
    for item, result, (ox, oy, sx, sy) in zip(keep, results, offsets):
        kp_obj = getattr(result, "keypoints", None)
        if kp_obj is None or kp_obj.xy is None or len(kp_obj.xy) == 0:
            continue
        kpts = kp_obj.xy[0].detach().cpu().numpy()
        confs = kp_obj.conf[0].detach().cpu().numpy() if kp_obj.conf is not None else [0.0] * len(kpts)
        # Map crop coords -> original frame coords.
        frame_kpts = []
        for (kx, ky), c in zip(kpts, confs):
            frame_kpts.append([float(kx) * sx + ox, float(ky) * sy + oy])
        item.detection.attributes["pose_keypoints"] = frame_kpts
        item.detection.attributes["pose_keypoint_conf"] = [float(c) for c in confs]


def _torso_crop_from_pose(frame: Any, item: Any, conf_threshold: float = 0.30) -> Any:
    """Compute a torso-tight crop from shoulder+hip pose keypoints. When a segmentation mask
    is available on the detection (YOLO11m-seg), grass/background pixels inside the torso
    rectangle are filled with the player's kit color so PARSeq isn't distracted by green
    pitch. Returns None if pose keypoints aren't available."""
    detection = getattr(item, "detection", None)
    if detection is None:
        return None
    kpts = detection.attributes.get("pose_keypoints")
    confs = detection.attributes.get("pose_keypoint_conf")
    if not kpts or not confs:
        return None
    idx_required = [5, 6, 11, 12]
    pts = []
    for idx in idx_required:
        if idx >= len(kpts) or idx >= len(confs):
            return None
        if confs[idx] < conf_threshold:
            return None
        pts.append(kpts[idx])
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x1 = min(xs); x2 = max(xs)
    y1 = min(ys); y2 = max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    w = x2 - x1; h = y2 - y1
    pad_x = 0.30 * w
    pad_y_top = 0.10 * h
    pad_y_bot = 0.10 * h
    fh, fw = frame.shape[:2]
    x1c = int(max(0, x1 - pad_x))
    x2c = int(min(fw, x2 + pad_x))
    y1c = int(max(0, y1 - pad_y_top))
    y2c = int(min(fh, y2 + pad_y_bot))
    if x2c - x1c < 12 or y2c - y1c < 12:
        return None
    crop = frame[y1c:y2c, x1c:x2c].copy()
    # Apply segmentation mask if available: wipe non-player (grass) pixels with kit color.
    seg_mask = detection.attributes.get("segmentation_mask")
    if seg_mask is not None:
        import numpy as _np
        mask_crop = seg_mask[y1c:y2c, x1c:x2c]
        if mask_crop.shape[:2] == crop.shape[:2] and mask_crop.any():
            # Estimate kit color from the player-mask pixels themselves (median where mask=1).
            kit_pixels = crop[mask_crop > 0]
            if kit_pixels.size > 0:
                kit_bgr = _np.median(kit_pixels.reshape(-1, 3), axis=0).astype(_np.uint8)
            else:
                kit_bgr = _np.array([0, 0, 0], dtype=_np.uint8)
            # Dilate mask by a few pixels so we don't clip the jersey edge.
            kernel = __import__("cv2").getStructuringElement(__import__("cv2").MORPH_ELLIPSE, (5, 5))
            mask_dilated = __import__("cv2").dilate(mask_crop, kernel, iterations=1)
            crop[mask_dilated == 0] = kit_bgr
    return crop


def _compute_idf_weights(
    tracklets: list[Tracklet],
    peak_threshold: float,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """Inverse-frequency calibration of jersey predictions.

    PARSeq has a strong "10" bias: on this clip, "10" appears as a peak read in many tracks
    (because the digit shape is simple), while "9" appears far fewer times. Without
    calibration, the noisy "10" reads on Argentine outfield tracks edge out Álvarez's real
    "9" reads in the cross-track dedup. IDF re-weighting boosts rare predictions and
    discounts noisy common ones.

    Formula: weight[jersey] = log((N + smoothing) / (n_tracks_with_jersey + smoothing)) + 1.

    A jersey predicted as peak in 5/100 tracks gets weight ~log(20)+1 = 4.0; one predicted
    in 80/100 tracks gets weight ~log(1.25)+1 = 1.22.
    """
    import math

    n_tracks_total = 0
    n_with_jersey: dict[str, int] = {}
    for tracklet in tracklets:
        peak_jerseys = set()
        for obs in tracklet.observations:
            for jersey, prob in obs.jersey_probs.items():
                contribution = prob * max(0.01, obs.jersey_quality)
                if contribution >= peak_threshold:
                    peak_jerseys.add(jersey)
        if peak_jerseys:
            n_tracks_total += 1
        for j in peak_jerseys:
            n_with_jersey[j] = n_with_jersey.get(j, 0) + 1
    weights: dict[str, float] = {}
    for jersey, count in n_with_jersey.items():
        weights[jersey] = math.log((n_tracks_total + smoothing) / (count + smoothing)) + 1.0
    return weights


def _gather_top_k_frame_contributions(
    tracklet: Tracklet,
    top_k_per_jersey: int = 24,
) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate per-track OCR votes using ONLY the top-K most confident frames for each
    jersey candidate. This is much more robust than summing across all observations:
    PARSeq's "10" bias fires on noisy/blurry frames at moderate confidence and floods the
    sum even when the actual player is a different number. By keeping only the K frames
    where each candidate hit highest confidence, we sample the player's "best look" at the
    jersey and the bias washes out — the actual jersey's top-K matches the model's
    intrinsic recognition rate, while the noisy-bias jersey's top-K hits a much lower
    ceiling.

    Returns (votes, peak_counts) computed only on the top-K samples per jersey.
    """
    # First pass: collect (contribution, jersey) per observation.
    per_jersey_contribs: dict[str, list[float]] = {}
    for obs in tracklet.observations:
        for jersey, prob in obs.jersey_probs.items():
            contribution = prob * max(0.01, obs.jersey_quality)
            per_jersey_contribs.setdefault(jersey, []).append(contribution)
    # Keep top-K per jersey, then aggregate.
    votes: dict[str, float] = {}
    peak_counts: dict[str, int] = {}
    for jersey, contribs in per_jersey_contribs.items():
        contribs.sort(reverse=True)
        top = contribs[:top_k_per_jersey]
        votes[jersey] = sum(top)
        peak_counts[jersey] = sum(1 for c in top if c >= 0.85)
    return votes, peak_counts


def _ranked_jersey_candidates(
    tracklet: Tracklet,
    min_confidence: float,
    peak_threshold: float = 0.85,
    peak_min_count: int = 4,
    longer_lock_min_count: int = 3,
    idf_weights: dict[str, float] | None = None,
    top_k_per_jersey: int = 24,
) -> list[tuple[str, float]]:
    """Return a ranked list of (jersey, score) candidates for this tracklet, best first.
    Uses the top-K-most-confident-frames-per-jersey aggregation (see _gather_top_k_frame_contributions)
    so PARSeq's bias toward common digits doesn't accumulate across noisy frames."""
    votes, peak_counts = _gather_top_k_frame_contributions(tracklet, top_k_per_jersey=top_k_per_jersey)
    if not votes:
        return []
    # IDF-weighted per-track scores.
    if idf_weights:
        cal_peaks = {j: peak_counts.get(j, 0) * idf_weights.get(j, 1.0) for j in peak_counts}
        cal_votes = {j: votes.get(j, 0.0) * idf_weights.get(j, 1.0) for j in votes}
    else:
        cal_peaks = {j: float(c) for j, c in peak_counts.items()}
        cal_votes = dict(votes)
    total = sum(cal_votes.values())
    if total <= 0:
        return []
    scored: list[tuple[str, float, float]] = []
    for jersey, v in cal_votes.items():
        peaks_cal = cal_peaks.get(jersey, 0.0)
        if peaks_cal >= longer_lock_min_count:
            share = v / total
            score = v * share * (1.0 + min(1.0, peaks_cal / max(1, peak_min_count)))
            scored.append((jersey, score, peaks_cal))
    if scored:
        scored.sort(key=lambda t: (t[2], t[1]), reverse=True)
        return [(j, s) for j, s, _ in scored]
    weighted_best = max(cal_votes, key=cal_votes.get)
    share = cal_votes[weighted_best] / total
    if share < min_confidence:
        return []
    return [(weighted_best, cal_votes[weighted_best] * share)]


def _best_jersey_for_summary(
    tracklet: Tracklet,
    min_confidence: float,
    peak_threshold: float = 0.85,
    peak_min_count: int = 4,
    longer_lock_min_count: int = 3,
    idf_weights: dict[str, float] | None = None,
    top_k_per_jersey: int = 24,
) -> tuple[str | None, float]:
    """Return (jersey, score) for this track. Aggregation uses only the top-K
    most-confident frames per jersey candidate, so noisy bias reads don't accumulate."""
    votes, peak_counts = _gather_top_k_frame_contributions(tracklet, top_k_per_jersey=top_k_per_jersey)
    if not votes:
        return None, 0.0
    # IDF re-weight before ranking.
    if idf_weights:
        votes = {j: v * idf_weights.get(j, 1.0) for j, v in votes.items()}
        peak_counts_cal = {j: peak_counts.get(j, 0) * idf_weights.get(j, 1.0) for j in peak_counts}
    else:
        peak_counts_cal = {j: float(c) for j, c in peak_counts.items()}
    total = sum(votes.values())
    if total <= 0:
        return None, 0.0
    weighted_best = max(votes, key=votes.get)
    peak_locked = [(j, c) for j, c in peak_counts_cal.items() if c >= longer_lock_min_count]
    if peak_locked:
        chosen, chosen_peaks = max(peak_locked, key=lambda jc: (jc[1], votes.get(jc[0], 0.0)))
        chosen_share = votes.get(chosen, 0.0) / total
        chosen_score = votes.get(chosen, 0.0) * chosen_share * (1.0 + min(1.0, chosen_peaks / max(1, peak_min_count)))
        return chosen, chosen_score
    share = votes[weighted_best] / total
    peaks_cal = peak_counts_cal.get(weighted_best, 0.0)
    score = votes[weighted_best] * share * (1.0 + min(1.0, peaks_cal / max(1, peak_min_count)))
    if share >= min_confidence:
        return weighted_best, score
    return None, score


def _deduplicate_jerseys(
    tracks: list[dict[str, Any]],
    team_rosters: dict[str, set[str]] | None = None,
    cascade_min_peaks: int = 4,
) -> None:
    """Iterative (team, jersey) -> track assignment that lets a losing track fall back
    to its next ranked jersey candidate. Uses 'jersey_candidates' (list of (jersey, score))
    populated by _tracks_summary; falls back to single best_jersey_guess if absent.

    For each (team, jersey) pair we keep only the highest-scoring track. Tracks that lose
    their top claim advance their candidate pointer and re-enter the assignment pool;
    iterations stop when assignments stabilize.
    """
    # Goalkeeper/referee tracks must not compete with outfield players for jersey slots.
    # Clear their jersey state before dedup so they don't steal a real player's number.
    for t in tracks:
        if t.get("role") in {"goalkeeper", "referee"}:
            t["best_jersey_guess"] = None
            t["jersey_candidates"] = []
            t["jersey_excluded_due_to_role"] = True

    # Determine team per track (argmax of team_probs).
    def team_of(t: dict[str, Any]) -> str | None:
        tp = t.get("team_probs") or {}
        return max(tp, key=tp.get) if tp else None

    def is_valid(team: str | None, jersey: str) -> bool:
        if not team_rosters:
            return True
        roster = team_rosters.get(team)
        if roster is None:
            return True  # unknown team -> let it through
        return jersey in roster

    # Snapshot pointer state: each track has an index into its ranked candidate list.
    candidate_idx: dict[str, int] = {t["track_id"]: 0 for t in tracks}
    final: dict[str, tuple[str | None, float] | None] = {t["track_id"]: None for t in tracks}
    for _iteration in range(8):  # 8 rounds of fallback is plenty in practice
        claims: dict[tuple[str | None, str], tuple[float, str]] = {}  # (team, jersey) -> (score, track_id)
        for t in tracks:
            tid = t["track_id"]
            cands: list[tuple[str, float]] = t.get("jersey_candidates") or []
            if not cands:
                final[tid] = None
                continue
            idx = candidate_idx.get(tid, 0)
            if idx >= len(cands):
                final[tid] = None
                continue
            # Advance past candidates that violate the per-team roster.
            team = team_of(t)
            peaks = t.get("jersey_peak_counts") or {}
            while idx < len(cands):
                cand_jersey = cands[idx][0]
                if not is_valid(team, cand_jersey):
                    idx += 1; continue
                # Cascade gate: if this is not the track's top candidate, require enough
                # peak-confidence reads on it. Avoids labeling a track as #24 because every
                # better-supported candidate was taken by another track.
                if idx > 0 and peaks.get(cand_jersey, 0) < cascade_min_peaks:
                    idx += 1; continue
                break
            if idx >= len(cands):
                candidate_idx[tid] = idx
                final[tid] = None
                continue
            candidate_idx[tid] = idx
            jersey, score = cands[idx]
            key = (team, jersey)
            existing = claims.get(key)
            if existing is None or score > existing[0]:
                if existing is not None:
                    # Existing claimant just got bumped; advance its pointer for next iteration.
                    candidate_idx[existing[1]] = candidate_idx.get(existing[1], 0) + 1
                claims[key] = (score, tid)
                final[tid] = (jersey, score)
            else:
                # We lose this jersey; advance our pointer for next iteration.
                candidate_idx[tid] = idx + 1
                final[tid] = None
        # Stability check: if no pointer advanced in this round, we're done.
        if all(candidate_idx.get(t["track_id"], 0) < max(1, len(t.get("jersey_candidates") or [])) for t in tracks):
            stable = True
            for t in tracks:
                tid = t["track_id"]
                expected = final[tid]
                cands = t.get("jersey_candidates") or []
                idx = candidate_idx.get(tid, 0)
                got = (cands[idx][0], cands[idx][1]) if idx < len(cands) else None
                if (expected is None) != (got is None):
                    stable = False; break
                if expected and got and expected[0] != got[0]:
                    stable = False; break
            if stable:
                break

    for t in tracks:
        tid = t["track_id"]
        prev_guess = t.get("best_jersey_guess")
        result = final.get(tid)
        if result is None:
            t["best_jersey_guess"] = None
            if prev_guess is not None:
                t["jersey_deduplicated"] = True
        else:
            new_jersey, new_score = result
            t["best_jersey_guess"] = new_jersey
            t["jersey_vote_score"] = round(float(new_score), 4)
            if prev_guess is not None and prev_guess != new_jersey:
                t["jersey_fallback_applied"] = True


if __name__ == "__main__":
    main()
