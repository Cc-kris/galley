"""Managed GA interruption-resilience tests (patch 1000).

Covers the three behaviours the PRD asks for, loaded from the shipped payload
source so the assertions track the code Galley actually runs:

1. request-level backoff follows a configurable tier schedule instead of a
   fixed 3*2^n curve capped at 30 s;
2. the turn-level guard waits out a backoff before re-sending the turn, honours
   ``turn_retry_max``, and gives up immediately when the user stops;
3. Goal Mode closes on a ``completion_signal`` in the state file, but only
   after at least two wakes.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
import textwrap
import time
import types
from pathlib import Path
from typing import Any, cast

import pytest

_MANAGED_GA_CODE = Path(__file__).resolve().parents[2] / "managed-ga" / "code"
if str(_MANAGED_GA_CODE) not in sys.path:
    sys.path.insert(0, str(_MANAGED_GA_CODE))

_PREVIOUS_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
sys.dont_write_bytecode = True
sys.modules.setdefault("requests", types.ModuleType("requests"))
urllib3_stub = types.ModuleType("urllib3")
urllib3_typed = cast(Any, urllib3_stub)
urllib3_typed.exceptions = types.SimpleNamespace(InsecureRetryWarning=Warning)
urllib3_typed.exceptions = types.SimpleNamespace(InsecureRequestWarning=Warning)
urllib3_typed.disable_warnings = lambda *_args, **_kwargs: None
urllib3_typed.connection = types.SimpleNamespace(
    HTTPConnection=type("HTTPConnection", (), {"request": lambda self, *a, **k: None})
)
sys.modules.setdefault("urllib3", urllib3_stub)

try:
    import llmcore  # type: ignore[import-not-found]  # noqa: E402
finally:
    sys.dont_write_bytecode = _PREVIOUS_DONT_WRITE_BYTECODE


# --------------------------------------------------------------------------
# helpers: pull a nested function / method out of the payload source
# --------------------------------------------------------------------------


def _nested_function(func: Any, name: str, extra_globals: dict[str, Any]) -> Any:
    """Compile a function defined inside *func* as a standalone callable."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            namespace = dict(extra_globals)
            exec(compile(ast.fix_missing_locations(module), f"<{name}>", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found inside {getattr(func, '__name__', func)}")


def _class_method(
    relative_path: str,
    class_name: str,
    method_name: str,
    extra_globals: dict[str, Any],
) -> Any:
    """Compile one method of a payload class without importing the module."""
    source = (_MANAGED_GA_CODE / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    module = ast.Module(body=[child], type_ignores=[])
                    namespace = dict(extra_globals)
                    exec(
                        compile(ast.fix_missing_locations(module), f"<{method_name}>", "exec"),
                        namespace,
                    )
                    return namespace[method_name]
    raise AssertionError(f"{class_name}.{method_name} not found in {relative_path}")


def _delay(sess: Any) -> Any:
    """Compile the payload's nested ``_delay`` with its closure bound."""
    return _nested_function(
        llmcore._stream_with_retry,
        "_delay",
        {
            "_DEFAULT_RETRY_SCHEDULE": llmcore._DEFAULT_RETRY_SCHEDULE,
            "sess": sess,
            "cap": float(getattr(sess, "max_retry_after", 60.0)),
        },
    )


class _Response:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


class _Session:
    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


# --------------------------------------------------------------------------
# 1. request-level backoff schedule
# --------------------------------------------------------------------------


def test_default_schedule_is_the_prd_tier_table() -> None:
    delay = _delay(_Session())

    assert [delay(_Response(), attempt) for attempt in range(7)] == [
        3.0,
        10.0,
        30.0,
        60.0,
        120.0,
        120.0,
        120.0,
    ]


def test_retry_after_header_still_wins_over_the_schedule() -> None:
    delay = _delay(_Session())

    assert delay(_Response({"retry-after": "7"}), 0) == 7.0
    # Beyond max_retry_after the request is not retried at all.
    assert delay(_Response({"retry-after": "600"}), 0) is None


def test_schedule_comes_from_model_config() -> None:
    delay = _delay(_Session(retry_schedule=[1.0, 2.0]))

    assert [delay(_Response(), attempt) for attempt in range(3)] == [1.0, 2.0, 2.0]


@pytest.mark.parametrize(
    "value",
    [None, [], "5", {"a": 1}, [0], [-1], ["x"], [1, None]],
)
def test_unusable_schedule_falls_back_to_default(value: Any) -> None:
    assert llmcore._normalize_retry_schedule(value) is None


def test_normalize_accepts_positive_numbers() -> None:
    assert llmcore._normalize_retry_schedule([3, "10", 30.5]) == [3.0, 10.0, 30.5]


# --------------------------------------------------------------------------
# 2. turn-level retry
# --------------------------------------------------------------------------


class _FakeTime:
    """Deterministic clock: sleep() advances time instead of blocking."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _Outcome:
    """Stand-in for agent_loop.StepOutcome (positional result dict + flags)."""

    def __init__(self, result: Any = None, **kwargs: Any) -> None:
        self.result = result
        self.should_exit = kwargs.pop("should_exit", False)
        self.next_prompt = kwargs.pop("next_prompt", None)
        self.__dict__.update(kwargs)


class _Handler:
    def __init__(self, session: Any, empty_ct: int = 0) -> None:
        self._empty_ct = empty_ct
        self._session = session

    def _llm_session(self) -> Any:
        return self._session


def _retry_or_exit(clock: _FakeTime) -> Any:
    return _class_method(
        "ga.py",
        "GenericAgentHandler",
        "_retry_or_exit",
        {"time": clock, "StepOutcome": _Outcome},
    )


def _drain(gen: Any) -> tuple[list[str], Any]:
    yields: list[str] = []
    while True:
        try:
            yields.append(next(gen))
        except StopIteration as stop:
            return yields, stop.value


def test_turn_retry_waits_then_resends_the_turn() -> None:
    clock = _FakeTime()
    handler = _Handler(_Session(turn_retry_max=6))
    method = _retry_or_exit(clock)

    yields, outcome = _drain(method(handler, "PROMPT"))

    assert handler._empty_ct == 1
    assert clock.slept, "the guard must actually wait before re-sending"
    assert sum(clock.slept) == pytest.approx(10.0)
    assert any("轮次级重试" in line for line in yields)
    assert outcome.should_exit is False
    assert outcome.next_prompt == "PROMPT"


def test_turn_retry_backoff_widens_with_each_strike() -> None:
    delays = []
    for strike in range(1, 6):
        clock = _FakeTime()
        method = _retry_or_exit(clock)
        _drain(method(_Handler(_Session(turn_retry_max=6), empty_ct=strike - 1), "PROMPT"))
        delays.append(sum(clock.slept))

    assert delays == [10.0, 30.0, 60.0, 120.0, 300.0]


def test_turn_retry_gives_up_at_the_configured_budget() -> None:
    clock = _FakeTime()
    method = _retry_or_exit(clock)

    yields, outcome = _drain(method(_Handler(_Session(turn_retry_max=3), empty_ct=2), "PROMPT"))

    assert yields == []
    assert clock.slept == []
    assert outcome.should_exit is True


def test_turn_retry_defaults_to_six_strikes_without_config() -> None:
    clock = _FakeTime()
    method = _retry_or_exit(clock)

    _, outcome = _drain(method(_Handler(_Session(), empty_ct=5), "PROMPT"))

    assert outcome.should_exit is True


def test_turn_retry_stops_waiting_when_the_user_aborts() -> None:
    clock = _FakeTime()

    class _StoppingSession(_Session):
        def should_stop(self) -> bool:
            return True

    method = _retry_or_exit(clock)
    yields, outcome = _drain(method(_Handler(_StoppingSession(turn_retry_max=6)), "PROMPT"))

    assert clock.slept == [], "no sleep may happen after the user asked to stop"
    assert outcome.should_exit is True
    assert any("轮次级重试" in line for line in yields)


def test_session_reads_the_retry_settings_from_model_config() -> None:
    """BaseSession surfaces the two keys so ga.py can read them off the session."""
    source = (_MANAGED_GA_CODE / "llmcore.py").read_text(encoding="utf-8")
    assert "self.retry_schedule = _normalize_retry_schedule(cfg.get('retry_schedule'))" in source
    assert "self.turn_retry_max = max(1, int(cfg.get('turn_retry_max') or 6))" in source


def test_real_tool_call_resets_the_empty_response_counter() -> None:
    """A successful tool call must clear the strike counter (cross-task leak)."""
    source = (_MANAGED_GA_CODE / "agent_loop.py").read_text(encoding="utf-8")
    dispatch = source.split("if tool_name == 'no_tool': pass", 1)[1].split(
        "handler.current_turn", 1
    )[0]
    assert "handler._empty_ct = 0" in dispatch


# --------------------------------------------------------------------------
# 3. Goal Mode completion signal
# --------------------------------------------------------------------------


def _load_goal_mode() -> Any:
    spec = importlib.util.spec_from_file_location(
        "managed_ga_goal_mode", _MANAGED_GA_CODE / "reflect" / "goal_mode.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _goal_state(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    state = {
        "objective": "ship it",
        "status": "running",
        "start_time": time.time(),
        "budget_seconds": 3_600,
        "turns_used": 0,
    }
    state.update(overrides)
    return state


def _write_state(tmp_path: Path, state: dict[str, Any]) -> Path:
    path = tmp_path / "goal_state.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


def test_goal_mode_closes_on_completion_signal(tmp_path: Path) -> None:
    module = _load_goal_mode()
    path = _write_state(
        tmp_path,
        _goal_state(
            tmp_path,
            turns_used=3,
            completion_signal={"verdict": "done", "evidence": ["pytest"]},
        ),
    )
    module.init({"goal_state": str(path)})

    assert module.check() == "/exit"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "done_completed"
    assert "end_time" in saved


def test_goal_mode_ignores_a_first_turn_completion_signal(tmp_path: Path) -> None:
    module = _load_goal_mode()
    path = _write_state(
        tmp_path,
        _goal_state(tmp_path, turns_used=1, completion_signal={"verdict": "done"}),
    )
    module.init({"goal_state": str(path)})

    prompt = module.check()

    assert prompt != "/exit"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "running"


def test_goal_mode_keeps_running_without_a_signal(tmp_path: Path) -> None:
    module = _load_goal_mode()
    path = _write_state(tmp_path, _goal_state(tmp_path))
    module.init({"goal_state": str(path)})

    prompt = module.check()

    assert "Goal Mode" in prompt
    assert str(path) in prompt, "the model must be told which file to write the signal into"


def test_goal_mode_still_stops_at_the_budget(tmp_path: Path) -> None:
    module = _load_goal_mode()
    path = _write_state(
        tmp_path, _goal_state(tmp_path, start_time=time.time() - 3_600, budget_seconds=60)
    )
    module.init({"goal_state": str(path)})

    prompt = module.check()

    assert "预算耗尽" in prompt
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "wrapping_up"


def test_continuation_prompt_documents_the_completion_protocol() -> None:
    module = _load_goal_mode()

    assert "completion_signal" in module.CONTINUATION_PROMPT
    assert "无法宣告完成" not in module.CONTINUATION_PROMPT
