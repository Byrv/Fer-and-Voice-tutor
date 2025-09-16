import os
import re
import queue
import threading
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, List, Optional

import boto3
from botocore.exceptions import ClientError
from playsound import playsound
from dotenv import load_dotenv


class PollySpeaker:
    """
    Simple Text-to-Speech wrapper around Amazon Polly built for tutors/assistants.

    Features:
    - speak_response(response): Extract text from string/dict/object (.content/.text/.message)
    - speak_text(text): Speak direct text
    - Async playback via a single worker thread so chunks play in order (no overlap)
    - Falls back to Aditi/standard engine if Kajal/neural isn't available in your region
    - Splits long text into chunks to avoid Polly request limits
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")  # keep punctuation

    def __init__(
        self,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        voice: str = "Kajal",
        engine: str = "neural",
        lang: str = "en-IN",
        chunk_chars: int = 1800,
        async_default: bool = True,
    ):
        """
        Args:
            region: AWS region (defaults to AWS_DEFAULT_REGION or us-east-1).
            access_key/secret_key: If omitted, uses default AWS credential chain.
            voice: Preferred Polly voice (default 'Kajal' for en-IN neural).
            engine: 'neural' or 'standard'.
            lang: LanguageCode passed to Polly (useful for bilingual voices).
            chunk_chars: Approx max characters per Polly request.
            async_default: If True, speak_* methods queue audio (non-blocking).
        """
        load_dotenv()
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.voice = voice
        self.engine = engine
        self.lang = lang
        self.chunk_chars = max(500, int(chunk_chars))  # keep reasonable minimum
        self.async_default = async_default

        self._polly = boto3.client(
            "polly",
            region_name=self.region,
            aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        # Async playback infra
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # -------------------- Public API --------------------

    def speak_response(self, response: Any, async_play: Optional[bool] = None) -> None:
        """
        Extract text from common response shapes and speak it.
        """
        text = self._extract_text(response)
        if not text:
            print("[Voice] Nothing to speak (empty response).")
            return
        self.speak_text(text, async_play=async_play)

    def speak_text(self, text: str, async_play: Optional[bool] = None) -> None:
        """
        Speak raw text. Chunks long text; either enqueues for async playback or plays synchronously.
        """
        mode_async = self.async_default if async_play is None else async_play
        chunks = list(self._chunk_text(text, max_len=self.chunk_chars))
        if not chunks:
            return

        if mode_async:
            self._ensure_worker()
            for c in chunks:
                self._q.put(c)
        else:
            for c in chunks:
                self._synthesize_and_play_blocking(c)

    def stop(self):
        """
        Stop the async worker gracefully and drain the queue.
        """
        if self._worker and self._worker.is_alive():
            self._stop_event.set()
            # Send sentinel to unblock queue.get
            self._q.put(None)  # type: ignore
            self._worker.join(timeout=5)
        self._worker = None
        with self._q.mutex:
            self._q.queue.clear()
        self._stop_event.clear()

    # Context manager support
    def __enter__(self):
        self._ensure_worker()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # -------------------- Internals --------------------

    def _ensure_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._player_worker, daemon=True)
        self._worker.start()

    def _player_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:  # sentinel
                break
            try:
                self._synthesize_and_play_blocking(item)
            except Exception as e:
                print(f"[Voice] Playback error: {e}")
            finally:
                self._q.task_done()

    def _synthesize_and_play_blocking(self, text: str):
        # If you pass SSML like <speak>...</speak>, you can detect and set TextType="ssml"
        text_type = "ssml" if text.lstrip().startswith("<speak>") else "text"

        # First try preferred voice/engine
        try:
            resp = self._polly.synthesize_speech(
                Text=text,
                TextType=text_type,
                OutputFormat="mp3",
                VoiceId=self.voice,
                Engine=self.engine,
                LanguageCode=self.lang,
            )
        except ClientError as e:
            # Fallback to Aditi/standard (widely available for en-IN/hi-IN)
            print(f"[Polly] {e}. Falling back to Aditi/standard.")
            resp = self._polly.synthesize_speech(
                Text=text,
                TextType=text_type,
                OutputFormat="mp3",
                VoiceId="Aditi",
                Engine="standard",
                LanguageCode=self.lang,
            )

        data = resp["AudioStream"].read()

        # Write to a temp file and play (Windows-friendly: close before playing)
        tmp = NamedTemporaryFile(suffix=".mp3", delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            playsound(tmp.name)  # blocking play for one chunk
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """
        Best-effort extraction from str | dict | object(.content/.text/.message).
        """
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            for key in ("content", "text", "message", "answer"):
                val = resp.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            # common nested shapes
            out = resp.get("output")
            if isinstance(out, dict):
                t = out.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    t = first.get("text")
                    if isinstance(t, str) and t.strip():
                        return t.strip()
            return ""
        # object-like
        for attr in ("content", "text", "message"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        # last resort
        return str(resp).strip()

    @classmethod
    def _chunk_text(cls, text: str, max_len: int = 1800) -> Iterable[str]:
        """
        Chunk on sentence boundaries up to ~max_len characters.
        """
        sentences = re.split(cls._SENTENCE_SPLIT, text.strip())
        buf: List[str] = []
        length = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if length + len(s) + 1 > max_len and buf:
                yield " ".join(buf)
                buf = [s]
                length = len(s)
            else:
                buf.append(s)
                length += len(s) + 1
        if buf:
            yield " ".join(buf)
