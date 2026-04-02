# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
from .basetrack import TrackState
from .bot_sort import ReID
from .byte_tracker import STrack
from .oc_sort import OCSORT, OCSortTrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYAH


class DeepOCSortTrack(OCSortTrack):
    """Track object for Deep OC-SORT with appearance features and observation-centric state management.

    Extends OCSortTrack with ReID embedding storage and exponential moving average smoothing,
    plus confidence-adaptive embedding update rates.

    Attributes:
        smooth_feat (np.ndarray | None): Smoothed feature vector via EMA.
        curr_feat (np.ndarray | None): Current frame's feature vector.
        features (deque): Feature history buffer.
        alpha_fixed_emb (float): Base EMA factor for embedding updates.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(
        self,
        xywh: list[float],
        score: float,
        cls: Any,
        delta_t: int = 3,
        feat: np.ndarray | None = None,
        alpha_fixed_emb: float = 0.95,
        det_thresh: float = 0.25,
    ):
        """Initialize DeepOCSortTrack with appearance features."""
        super().__init__(xywh, score, cls, delta_t)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha_fixed_emb = alpha_fixed_emb
        self.det_thresh = det_thresh
        if feat is not None:
            self.update_features(feat, score)

    def update_features(self, feat: np.ndarray, score: float | None = None) -> None:
        """Update feature vector with confidence-adaptive EMA smoothing."""
        feat = feat / np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if score is not None and score > self.det_thresh:
                trust = (score - self.det_thresh) / (1 - self.det_thresh)
                alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)
            else:
                alpha = 1.0
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.smooth_feat = self.smooth_feat / np.linalg.norm(self.smooth_feat)

    def update(self, new_track: STrack, frame_id: int):
        """Update track state with a new matched detection, recording observation and features."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().update(new_track, frame_id)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a lost track with updated features."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().re_activate(new_track, frame_id, new_id)

    @staticmethod
    def multi_gmc(stracks: list, H: np.ndarray = np.eye(2, 3)):
        """Apply GMC correctly for XYAH state: rotate only (x,y) and (vx,vy), not (a,h)."""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]
        t = H[:2, 2]

        # Build 8x8 transform: rotate (x,y) and (vx,vy), identity for (a,h) and (va,vh)
        R8x8 = np.eye(8, dtype=float)
        R8x8[:2, :2] = R       # rotate position (x, y)
        R8x8[4:6, 4:6] = R     # rotate velocity (vx, vy)
        # indices 2,3 (a,h) and 6,7 (va,vh) remain identity

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            mean = R8x8.dot(mean)
            mean[:2] += t
            cov = R8x8.dot(cov).dot(R8x8.transpose())
            stracks[i].mean = mean
            stracks[i].covariance = cov

            # Also transform stored observations for OCR/ORU consistency
            if stracks[i].last_observation[0] >= 0:
                obs = stracks[i].last_observation
                # Transform xyxy observation centers
                cx, cy = (obs[0] + obs[2]) / 2, (obs[1] + obs[3]) / 2
                w, h = obs[2] - obs[0], obs[3] - obs[1]
                new_c = R @ np.array([cx, cy]) + t
                stracks[i].last_observation = np.array([
                    new_c[0] - w / 2, new_c[1] - h / 2,
                    new_c[0] + w / 2, new_c[1] + h / 2,
                ], dtype=np.float32)


class DeepOCSORT(OCSORT):
    """Deep OC-SORT: OC-SORT enhanced with appearance features, GMC, and adaptive weighting.

    Fixes over naive integration:
    - GMC correctly handles XYAH state (rotates only x,y positions, not aspect ratio/height)
    - Cost combination uses min(IoU, appearance) following BOTSORT's proven approach
    - OCR recovery pass also uses appearance features
    - ByteTrack-style low-confidence second pass enabled by default
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize Deep OC-SORT tracker."""
        super().__init__(args, frame_rate)

        # GMC for camera motion compensation
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))

        # Appearance parameters
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.75)
        self.alpha_fixed_emb = getattr(args, "alpha_fixed_emb", 0.95)

        # ReID encoder
        self.with_reid = getattr(args, "with_reid", False)
        if self.with_reid:
            model = getattr(args, "model", "auto")
            if model == "auto":
                self.encoder = lambda feats, s: [f.cpu().numpy() for f in feats]
            else:
                self.encoder = ReID(model)
        else:
            self.encoder = None

    def init_track(self, results, img=None):
        """Initialize DeepOCSortTrack instances with optional appearance features."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

        if self.with_reid and self.encoder is not None:
            features = self.encoder(img, bboxes)
            return [
                DeepOCSortTrack(
                    xywh, s, c, self.delta_t, feat=f,
                    alpha_fixed_emb=self.alpha_fixed_emb,
                    det_thresh=self.args.track_high_thresh,
                )
                for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features)
            ]
        return [
            DeepOCSortTrack(xywh, s, c, self.delta_t, alpha_fixed_emb=self.alpha_fixed_emb)
            for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
        ]

    def get_dists(self, tracks, detections):
        """Compute cost matrix combining Buffered IoU and appearance (BOTSORT-style min fusion).

        Uses min(BIoU_cost, appearance_cost) with proximity gating, plus OCM velocity cost.
        """
        iou_dists = self._biou_distance(tracks, detections)
        dists_mask = iou_dists > (1 - self.proximity_thresh)

        if self.args.fuse_score:
            dists = matching.fuse_score(iou_dists, detections)
        else:
            dists = iou_dists.copy()

        # Add OCM velocity direction consistency cost
        vel_dists = self._velocity_direction_cost(tracks, detections)
        dists = dists + self.inertia * vel_dists

        # Appearance: min fusion with strict gating
        if self.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)

        return dists

    def update(self, results, img=None, feats=None):
        """Update tracker with Deep OC-SORT pipeline: GMC + ReID + ORU/OCM/OCR + ByteTrack."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]

        # Handle features for ReID
        use_native_feats = self.with_reid and self.encoder is not None and getattr(self.args, "model", "auto") == "auto"
        feats_keep = feats_second = img
        if use_native_feats and feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep if use_native_feats else img)

        # Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Stage 1: First association with high-score detections
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        # Apply GMC with XYAH-correct transform
        if img is not None:
            try:
                warp = self.gmc.apply(img, results.xyxy if len(results) else np.empty((0, 4)))
            except Exception:
                warp = np.eye(2, 3)
            DeepOCSortTrack.multi_gmc(strack_pool, warp)
            DeepOCSortTrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.apply_oru(det.xyxy, self.frame_id)
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # OCR: Observation-Centric Recovery with appearance features
        ocr_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        ocr_dets = [detections[i] for i in u_detection]

        if ocr_tracked and ocr_dets:
            # IoU from last observation position
            ocr_dists = self._ocr_distance(ocr_tracked, ocr_dets)
            if self.args.fuse_score:
                ocr_dists = matching.fuse_score(ocr_dists, ocr_dets)

            # Enhance OCR with appearance (if ReID available)
            if self.with_reid and self.encoder is not None:
                ocr_emb_dists = matching.embedding_distance(ocr_tracked, ocr_dets) / 2.0
                ocr_emb_dists[ocr_emb_dists > (1 - self.appearance_thresh)] = 1.0
                ocr_dists = np.minimum(ocr_dists, ocr_emb_dists)

            ocr_matches, ocr_u_track, ocr_u_det = matching.linear_assignment(ocr_dists, thresh=self.args.match_thresh)

            for itracked, idet in ocr_matches:
                track = ocr_tracked[itracked]
                det = ocr_dets[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.apply_oru(det.xyxy, self.frame_id)
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            ocr_u_track_set = {id(ocr_tracked[i]) for i in ocr_u_track}
            ocr_u_det_set = {id(ocr_dets[i]) for i in ocr_u_det}
            u_track = [
                i for i in u_track
                if id(strack_pool[i]) in ocr_u_track_set or strack_pool[i].state != TrackState.Tracked
            ]
            u_detection = [i for i in u_detection if id(detections[i]) in ocr_u_det_set]

        # Stage 2: Low-confidence second pass (ByteTrack-style)
        if self.use_byte:
            detections_second = self.init_track(results_second, feats_second if use_native_feats else img)
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections_second)
            matches, u_track_second, _ = matching.linear_assignment(dists, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            for it in u_track_second:
                track = r_tracked_stracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            for i in u_track:
                track = strack_pool[i]
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)

        # Stage 3: Unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Stage 4: Init new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Stage 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def reset(self):
        """Reset the Deep OC-SORT tracker."""
        super().reset()
        self.gmc.reset_params()
