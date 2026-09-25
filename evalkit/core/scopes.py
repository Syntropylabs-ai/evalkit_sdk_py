from __future__ import annotations

import functools
import inspect
from contextvars import Token
from typing import Any, Callable, List, Optional, TypeVar

from .context import reset_identity, set_identity

F = TypeVar("F", bound=Callable[..., Any])


class session:
    """Scope every span opened inside it to one conversation.

    ``with evalkit.session("sess_1", user_id="u_1"):`` or ``@evalkit.session("sess_1")``. The
    identity in force before the block is restored on exit, so nested and concurrent scopes
    (threads, asyncio tasks) never see each other's session. A span that names its own session
    (argument or attribute) or sits under a span that does keeps that one.
    """

    def __init__(self, session_id: str, user_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self.user_id = user_id
        # A stack rather than one slot so the same instance can be re-entered.
        self._tokens: List[Token] = []

    def __enter__(self) -> "session":
        self._tokens.append(set_identity(user_id=self.user_id, session_id=self.session_id))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tokens:
            reset_identity(self._tokens.pop())

    async def __aenter__(self) -> "session":
        return self.__enter__()

    async def __aexit__(self, *exc: object) -> None:
        self.__exit__(*exc)

    def __call__(self, fn: F) -> F:
        # Each call opens its own scope: a decorator instance is shared by every concurrent call.
        session_id, user_id = self.session_id, self.user_id

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with session(session_id, user_id=user_id):
                    return await fn(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with session(session_id, user_id=user_id):
                return fn(*args, **kwargs)
        return sync_wrapper  # type: ignore[return-value]
