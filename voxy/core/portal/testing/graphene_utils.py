from dataclasses import dataclass
from typing import Optional

from django.contrib.auth.models import User


@dataclass
class GrapheneContext:
    user: Optional[User] = None
