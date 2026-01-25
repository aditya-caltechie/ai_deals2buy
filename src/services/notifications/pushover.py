from __future__ import annotations

from dataclasses import dataclass

import requests


PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


@dataclass
class PushoverClient:
    user: str
    token: str

    def send(self, message: str, *, sound: str = "cashregister") -> None:
        payload = {
            "user": self.user,
            "token": self.token,
            "message": message,
            "sound": sound,
        }
        requests.post(PUSHOVER_URL, data=payload)

