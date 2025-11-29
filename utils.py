import math
import threading
import simpleaudio as sa
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def eye_aspect_ratio(eye_landmarks):
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

class AlarmPlayer:
    def __init__(self, alarm_wav_path=None, fallback_freq=880, fallback_duration=0.3):
        self.alarm_wav_path = alarm_wav_path
        self.fallback_freq = fallback_freq
        self.fallback_duration = fallback_duration
        self._playing = False
        self._thread = None

    def _play_wav(self, path):
        try:
            wave_obj = sa.WaveObject.from_wave_file(path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except:
            self._beep()

    def _beep(self):
        fs = 44100
        t = np.linspace(0, self.fallback_duration, int(fs * self.fallback_duration), False)
        tone = np.sin(self.fallback_freq * t * 2 * np.pi)
        audio = tone * (2**15 - 1) / np.max(np.abs(tone))
        audio = audio.astype(np.int16)
        try:
            sa.play_buffer(audio, 1, 2, fs).wait_done()
        except:
            pass

    def _play_loop(self):
        while self._playing:
            if self.alarm_wav_path:
                self._play_wav(self.alarm_wav_path)
            else:
                self._beep()

    def start(self):
        if self._playing:
            return
        self._playing = True
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._playing = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None
