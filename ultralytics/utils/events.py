# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from ultralytics import SETTINGS, __version__
from ultralytics.utils import ARGV, ENVIRONMENT, GIT, IS_PIP_PACKAGE, ONLINE, PYTHON_VERSION, RANK, TESTS_RUNNING
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
from ultralytics.utils.torch_utils import get_cpu_info


def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    """Send a one-shot JSON POST request."""
    try:
        body = json.dumps(data, separators=(",", ":")).encode()  # compact JSON
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass


class Events:
    """Collect and send anonymous usage analytics with rate-limiting.

    Event collection and transmission are enabled when sync is enabled in settings, the current process is rank -1 or 0,
    tests are not running, the environment is online, and the installation source is either pip or the official
    Ultralytics GitHub repository.

    Attributes:
        url (str): Measurement Protocol endpoint for receiving anonymous events.
        events (list[dict]): In-memory queue of event payloads awaiting transmission.
        rate_limit (float): Minimum time in seconds between POST requests.
        t (float): Timestamp of the last transmission in seconds since the epoch.
        metadata (dict): Static metadata describing runtime, installation source, and environment.
        enabled (bool): Flag indicating whether analytics collection is active.

    Methods:
        __init__: Initialize the event queue, rate limiter, and runtime metadata.
        __call__: Queue an event and trigger a non-blocking send when the rate limit elapses.
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self) -> None:
        """Initialize the Events instance with queue, rate limiter, and environment metadata."""
        self.events = []  # pending events
        self.rate_limit = 30.0  # rate limit (seconds)
        self.t = 0.0  # last send timestamp (seconds)
        self._thread = None  # reference to the last background send thread
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],  # i.e. 3.13
            "CPU": get_cpu_info(),
            # "GPU": get_gpu_info(index=0) if cuda else None,
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
            "debug_mode":True,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and not TESTS_RUNNING
            and ONLINE
            and (IS_PIP_PACKAGE or GIT.origin == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg, device=None, backend=None, imgsz=None, model_params=None, speed=None) -> None:
        """Queue an event and flush the queue asynchronously when the rate limit elapses.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
            device (torch.device | str, optional): The device type (e.g., 'cpu', 'cuda').
            backend (object | None, optional): The inference backend instance used during prediction.
            imgsz (int | list | None, optional): Input image size used during prediction.
            model_params (int | None, optional): Total number of model parameters.
            speed (dict | None, optional): Per-image inference speed dict with keys 'preprocess',
                'inference', and 'postprocess' (all in milliseconds).
        """
        if not self.enabled:
            # Events disabled, do nothing
            return

        # Attempt to enqueue a new event
        if len(self.events) < 25:  # Queue limited to 25 events to bound memory and traffic
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            if cfg.mode == "predict":
                params["backend"] = type(backend).__name__ if backend is not None else None
                if imgsz is not None:
                    params["imgsz"] = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
                if model_params is not None:
                    params["model_params"] = model_params
                if speed is not None:
                    params["speed_preprocess_ms"] = round(speed.get("preprocess") or 0, 2)
                    params["speed_inference_ms"] = round(speed.get("inference") or 0, 2)
                    params["speed_postprocess_ms"] = round(speed.get("postprocess") or 0, 2)
            self.events.append({"name": cfg.mode, "params": params})

        # Check rate limit and return early if under limit
        t = time.time()
        if (t - self.t) < self.rate_limit:
            return

        # Overrate limit: send a snapshot of queued events in a background thread
        payload_events = list(self.events)  # snapshot to avoid race with queue reset
        self._thread = Thread(
            target=_post,
            args=(self.url, {"client_id": SETTINGS["uuid"], "events": payload_events}),  # SHA-256 anonymized
            daemon=True,
        )
        self._thread.start()

        # Reset queue and rate limit timer
        self.events = []
        self.t = t

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the in-flight background send thread completes or times out.

        Call this at the end of short-lived processes (e.g. single-image predict) so the
        daemon thread is not killed before the POST request finishes.

        Args:
            timeout (float): Maximum seconds to wait for the thread to finish.
        """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


events = Events()
