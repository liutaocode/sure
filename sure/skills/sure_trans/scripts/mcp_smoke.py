#!/usr/bin/env python3
"""Deterministic MCP protocol smoke driver for the SURE-TRANS adapter.

Runs inside the adapter image. Spawns the adapter MCP server and drives the
stdin/stdout JSON-RPC protocol with bounded deadlines: initialize,
tools/list, tools/call, shutdown. Every read is deadline-bounded and the
server process is killed on timeout, so this script always terminates and
always writes --produces evidence (even on failure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

SCHEMA = "sure.trans.mcp_smoke.v1"
STDERR_TAIL_LINES = 200
PORTABLE_CONTAINER_ROOTS = ("/fixture/", "/models/", "/opt/", "/validation/", "/workspace/")
KWS_OPERATING_THRESHOLD = 0.5


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_server_command(command: list[str]) -> list[str]:
    portable: list[str] = []
    for argument in command:
        if argument.startswith(PORTABLE_CONTAINER_ROOTS) or not Path(argument).is_absolute():
            portable.append(argument)
        else:
            portable.append(Path(argument).name)
    return portable


def tool_arguments(
    tool: str,
    audio: Path,
    fixture_row: dict | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    if tool == "synthesize_speech":
        return {
            "text": "SURE smoke test",
            "prompt_audio_path": str(audio),
        }
    if tool == "convert_voice":
        return {
            "source_audio_path": str(audio),
            "reference_audio_path": str(audio),
        }
    if tool == "kws_predict":
        arguments: dict[str, object] = {"audio_path": str(audio)}
        if fixture_row is not None and isinstance(fixture_row.get("keywords"), (str, list)):
            arguments["keywords"] = fixture_row["keywords"]
        if fixture_row is not None:
            threshold = fixture_row.get("threshold")
            if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                if not math.isfinite(float(threshold)) or float(threshold) != KWS_OPERATING_THRESHOLD:
                    raise ValueError(f"KWS fixture threshold must equal {KWS_OPERATING_THRESHOLD}")
                arguments["threshold"] = threshold
            elif threshold is not None:
                raise ValueError(f"KWS fixture threshold must equal {KWS_OPERATING_THRESHOLD}")
        return arguments
    if tool == "enhance_speech":
        arguments = {"audio_path": str(audio)}
        if output_path is not None:
            arguments["output_path"] = str(output_path)
        return arguments
    return {"audio_path": str(audio)}


def primary_output_field(tool: str) -> str:
    if tool in {"synthesize_speech", "convert_voice", "enhance_speech"}:
        return "audio_path"
    if tool == "kws_predict":
        return "detected"
    return "text"


def output_is_nonempty(primary_field: str, value: object) -> bool:
    """Whether the tool really produced its primary output.

    A path-valued field only proves an output exists if the file is there and
    holds bytes; the server runs as this script's child, so the path it
    returns is one this process can stat.
    """
    if primary_field.endswith("_path"):
        if not isinstance(value, str) or not value:
            return False
        candidate = Path(value)
        return candidate.is_file() and candidate.stat().st_size > 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def normalized_keyword(value: str) -> str:
    return "".join(value.upper().split())


def expected_detected(row: dict) -> bool:
    value = row.get("expected_detected")
    if isinstance(value, bool):
        return value
    value = row.get("expected", row.get("label"))
    normalized = str(value).strip().lower()
    if normalized in {"detect", "detected", "positive", "true", "1", "yes"}:
        return True
    if normalized in {"reject", "rejected", "negative", "false", "0", "no"}:
        return False
    raise ValueError("KWS fixture row requires an explicit positive or negative annotation")


def expected_keyword(row: dict, detected: bool) -> str | None:
    value = row.get("expected_keyword")
    if value is None and detected:
        value = row.get("text", row.get("txt"))
    if detected:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("positive KWS fixture row requires expected_keyword or text")
        return value.strip()
    if value not in (None, ""):
        raise ValueError("negative KWS fixture row must not declare expected_keyword")
    return None


def validate_kws_output(value: object, reference: dict) -> list[str]:
    if not isinstance(value, dict):
        return ["KWS output must be an object"]
    violations: list[str] = []
    for field in ("detected", "keyword", "score"):
        if field not in value:
            violations.append(f"missing required field: {field}")
    detected = value.get("detected")
    keyword = value.get("keyword")
    score = value.get("score")
    if not isinstance(detected, bool):
        violations.append("detected must be a boolean")
    if keyword is not None and (not isinstance(keyword, str) or not keyword.strip()):
        violations.append("keyword must be a non-empty string or null")
    if score is not None and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        violations.append("score must be a finite number or null")
    elif isinstance(score, (int, float)) and not isinstance(score, bool) and not 0 <= float(score) <= 1:
        violations.append("score must be within [0, 1]")
    if detected is True:
        if not isinstance(keyword, str) or not keyword.strip():
            violations.append("detected=true requires a keyword")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            violations.append("detected=true requires a finite numeric score")
        elif float(score) < KWS_OPERATING_THRESHOLD:
            violations.append(f"detected=true requires score >= {KWS_OPERATING_THRESHOLD}")
    elif detected is False and keyword is not None:
        violations.append("detected=false requires keyword=null")
    elif (
        detected is False
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        and float(score) >= KWS_OPERATING_THRESHOLD
    ):
        violations.append(f"detected=false requires score < {KWS_OPERATING_THRESHOLD}")
    reference_detected = expected_detected(reference)
    if isinstance(detected, bool) and detected is not reference_detected:
        violations.append(
            f"detection disagrees with fixture: expected {reference_detected}, got {detected}"
        )
    reference_keyword = expected_keyword(reference, reference_detected)
    if reference_detected and isinstance(keyword, str) and reference_keyword is not None:
        if normalized_keyword(keyword) != normalized_keyword(reference_keyword):
            violations.append(
                f"keyword disagrees with fixture: expected {reference_keyword!r}, got {keyword!r}"
            )
    return violations


def load_kws_fixture(path: Path) -> list[tuple[str, Path, dict]]:
    rows: list[tuple[str, Path, dict]] = []
    seen: set[str] = set()
    polarities: set[bool] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"KWS fixture line {line_number} must be an object")
        key = str(row.get("key") or "").strip()
        if not key or key in seen:
            raise ValueError(f"KWS fixture key is missing or duplicated: {key!r}")
        seen.add(key)
        audio = row.get("audio") or row.get("wav")
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"KWS fixture {key} requires audio or wav")
        relative = Path(audio)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"KWS fixture {key} audio path must be relative and contained")
        audio_path = (path.parent / relative).resolve()
        if not audio_path.is_file() or not audio_path.is_relative_to(path.parent.resolve()):
            raise ValueError(f"KWS fixture {key} audio is missing or unsafe")
        polarities.add(expected_detected(row))
        rows.append((key, audio_path, row))
    if not 2 <= len(rows) <= 5 or polarities != {False, True}:
        raise ValueError("KWS MCP smoke requires 2 to 5 samples with positive and negative coverage")
    return rows


def load_se_fixture(path: Path) -> list[tuple[str, Path, dict, Path]]:
    rows: list[tuple[str, Path, dict, Path]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"SE fixture line {line_number} must be an object")
        key = str(row.get("sample_id") or row.get("key") or "").strip()
        if not key or key in seen:
            raise ValueError(f"SE fixture key is missing or duplicated: {key!r}")
        seen.add(key)
        paths: dict[str, Path] = {}
        for role in ("noisy_audio", "reference_audio"):
            value = row.get(role, row.get("audio")) if role == "noisy_audio" else row.get(role)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SE fixture {key} requires {role}")
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"SE fixture {key} {role} path must be relative and contained")
            audio_path = path.parent / relative
            if (
                audio_path.is_symlink()
                or not audio_path.is_file()
                or not audio_path.resolve().is_relative_to(path.parent.resolve())
            ):
                raise ValueError(f"SE fixture {key} {role} is missing or unsafe")
            paths[role] = audio_path.resolve()
        if paths["noisy_audio"].samefile(paths["reference_audio"]):
            raise ValueError(
                f"SE fixture {key} noisy_audio and reference_audio must be independent files"
            )
        rows.append((key, paths["noisy_audio"], row, paths["reference_audio"]))
    if not 1 <= len(rows) <= 5:
        raise ValueError("SE MCP smoke requires 1 to 5 noisy/clean samples")
    return rows


def mcp_output_path(produces: Path, key: str, index: int) -> Path:
    root = produces.parent / "outputs"
    if root.is_symlink():
        raise ValueError("MCP outputs directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise ValueError("MCP outputs directory must be writable")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return root / f"mcp-{index:02d}-{digest}.wav"


def validate_se_output(
    value: object,
    *,
    key: str,
    outputs_root: Path,
    expected_path: Path,
    forbidden_inputs: tuple[Path, ...],
) -> tuple[list[str], Path | None]:
    if not isinstance(value, dict):
        return ["SE output must be an object"], None
    raw_path = value.get("audio_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ["SE output requires audio_path"], None
    candidate = Path(raw_path)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size <= 0:
        return ["SE audio_path must name a real non-empty file"], None
    root = outputs_root.resolve()
    try:
        lexical_relative = candidate.absolute().relative_to(root)
    except ValueError:
        return ["SE audio_path must stay below MCP validation outputs"], None
    current = root
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            return ["SE audio_path must not traverse a symlink"], None
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        return ["SE audio_path must stay below MCP validation outputs"], None
    if candidate.absolute() != expected_path.absolute() or resolved != expected_path.resolve():
        return ["SE audio_path must equal the harness-assigned output_path"], None
    for input_path in forbidden_inputs:
        try:
            aliases_input = resolved.samefile(input_path)
        except OSError:
            aliases_input = False
        if aliases_input:
            return ["SE audio_path must not alias noisy or clean input audio"], None
    try:
        with wave.open(str(resolved), "rb") as handle:
            if (
                handle.getcomptype() != "NONE"
                or handle.getnchannels() < 1
                or handle.getsampwidth() not in {1, 2, 3, 4}
                or handle.getframerate() < 1
                or handle.getnframes() < 1
            ):
                return ["SE audio_path must be a non-empty PCM WAV"], None
    except (EOFError, OSError, wave.Error):
        return ["SE audio_path must be a readable PCM WAV"], None
    return [], resolved


def _read_line(fd: int, buffer: bytearray, deadline: float) -> str | None:
    """Read one line from fd before deadline; None on timeout or EOF."""
    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            line = bytes(buffer[: newline + 1])
            del buffer[: newline + 1]
            return line.decode("utf-8", errors="replace").rstrip("\n")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        if not ready:
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            return None
        buffer.extend(chunk)


def _drain_stderr(proc: subprocess.Popen, tail: deque, log_handle) -> None:
    if proc.stderr is None:
        return
    for raw in proc.stderr:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        tail.append(line)
        while len(tail) > STDERR_TAIL_LINES:
            tail.popleft()
        if log_handle is not None:
            log_handle.write(line + "\n")
            log_handle.flush()


def _send(proc: subprocess.Popen, request: dict, deadline: float) -> bool:
    if proc.stdin is None:
        return False
    payload = json.dumps(request, ensure_ascii=False) + "\n"
    try:
        proc.stdin.write(payload.encode("utf-8"))
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        return False
    return time.monotonic() <= deadline


def _read_response(
    fd: int, buffer: bytearray, expected_id: int, deadline: float, junk_tail: deque[str]
) -> tuple[bool, dict]:
    while True:
        line = _read_line(fd, buffer, deadline)
        if line is None:
            return False, {}
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            junk_tail.append(line)
            continue
        if not isinstance(payload, dict) or payload.get("id") != expected_id:
            return False, payload
        if "error" in payload:
            return False, payload
        return "result" in payload, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive the adapter MCP JSON-RPC protocol with bounded deadlines.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio")
    inputs.add_argument("--fixture-gt-jsonl")
    parser.add_argument("--tool", default="transcribe_audio")
    parser.add_argument("--server-command", nargs="*", default=["python", "/opt/sure_trans/server.py"])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--produces", required=True)
    parser.add_argument("--server-stderr-log")
    args = parser.parse_args()

    audio = Path(args.audio) if args.audio else None
    fixture_gt_jsonl = Path(args.fixture_gt_jsonl) if args.fixture_gt_jsonl else None
    produces = Path(args.produces)
    steps: dict = {
        "initialize": {"ok": False},
        "tools_list": {"ok": False},
        "tools_call": {"ok": False, "output_nonempty": False, "text_nonempty": False},
        "shutdown": {"ok": False},
    }
    server_stderr: deque[str] = deque()
    junk_tail: deque[str] = deque()
    error: str | None = None
    started = time.monotonic()
    deadline = started + args.timeout

    stderr_handle = None
    if args.server_stderr_log:
        try:
            stderr_handle = open(args.server_stderr_log, "a", encoding="utf-8")
        except OSError:
            stderr_handle = None

    proc: subprocess.Popen | None = None
    try:
        if fixture_gt_jsonl is not None:
            if args.tool == "kws_predict":
                calls = [(*call, None) for call in load_kws_fixture(fixture_gt_jsonl)]
            elif args.tool == "enhance_speech":
                calls = load_se_fixture(fixture_gt_jsonl)
            else:
                raise ValueError(
                    "--fixture-gt-jsonl is only valid with --tool kws_predict or enhance_speech"
                )
        else:
            assert audio is not None
            calls = [(audio.stem, audio, None, None)]
        proc = subprocess.Popen(
            args.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=_drain_stderr, args=(proc, server_stderr, stderr_handle), daemon=True).start()
        stdout_buffer = bytearray()

        ok, payload = False, {}
        if _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, deadline):
            ok, payload = _read_response(proc.stdout.fileno(), stdout_buffer, 1, deadline, junk_tail)
        if ok:
            steps["initialize"] = {
                "ok": True,
                "protocolVersion": str((payload.get("result") or {}).get("protocolVersion") or ""),
            }
        else:
            raise RuntimeError(f"initialize step failed: {json.dumps(payload, ensure_ascii=False)[:500]}")

        if _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, deadline):
            ok, payload = _read_response(proc.stdout.fileno(), stdout_buffer, 2, deadline, junk_tail)
        tools = []
        if ok:
            tools = [item.get("name") for item in (payload.get("result") or {}).get("tools", []) if isinstance(item, dict)]
        steps["tools_list"] = {"ok": ok and args.tool in tools, "tools": tools}
        if not steps["tools_list"]["ok"]:
            raise RuntimeError(f"tools/list step failed for tool {args.tool!r}: tools={tools}")

        primary_field = primary_output_field(args.tool)
        sample_evidence: list[dict] = []
        all_calls_ok = True
        for sample_index, (key, call_audio, fixture_row, reference_audio) in enumerate(calls, 1):
            request_id = sample_index + 2
            requested_output = (
                mcp_output_path(produces, key, sample_index)
                if args.tool == "enhance_speech"
                else None
            )
            if requested_output is not None and (requested_output.exists() or requested_output.is_symlink()):
                requested_output.unlink()
            ok, payload = False, {}
            if _send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {
                        "name": args.tool,
                        "arguments": tool_arguments(
                            args.tool,
                            call_audio,
                            fixture_row,
                            requested_output,
                        ),
                    },
                },
                deadline,
            ):
                ok, payload = _read_response(
                    proc.stdout.fileno(), stdout_buffer, request_id, deadline, junk_tail
                )
            parsed: object = None
            if ok:
                try:
                    content = (payload.get("result") or {}).get("content") or []
                    parsed = json.loads(str(content[0].get("text") or ""))
                except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                    parsed = None
            primary_value = parsed.get(primary_field) if isinstance(parsed, dict) else None
            output_nonempty = output_is_nonempty(primary_field, primary_value)
            violations = (
                validate_kws_output(parsed, fixture_row)
                if args.tool == "kws_predict" and fixture_row is not None
                else []
            )
            generated_audio: Path | None = None
            if args.tool == "enhance_speech":
                se_violations, generated_audio = validate_se_output(
                    parsed,
                    key=key,
                    outputs_root=produces.parent / "outputs",
                    expected_path=requested_output,
                    forbidden_inputs=tuple(
                        path for path in (call_audio, reference_audio) if path is not None
                    ),
                )
                violations.extend(se_violations)
            call_ok = ok and output_nonempty and not violations
            all_calls_ok = all_calls_ok and call_ok
            result_preview: object = (
                {
                    field: parsed.get(field)
                    for field in ("detected", "keyword", "score")
                    if field in parsed
                }
                if isinstance(parsed, dict) and args.tool == "kws_predict"
                else primary_value
            )
            if args.tool == "enhance_speech":
                result_preview = (
                    {
                        "audio_path": f"outputs/{generated_audio.name}",
                        "audio_sha256": sha256_file(generated_audio),
                    }
                    if generated_audio is not None
                    else None
                )
            sample_evidence.append(
                {
                    "key": key,
                    "audio": (
                        str(
                            fixture_row.get("noisy_audio")
                            or fixture_row.get("audio")
                            or fixture_row.get("wav")
                        )
                        if fixture_row is not None
                        else call_audio.name
                    ),
                    "audio_sha256": sha256_file(call_audio),
                    "reference_audio": (
                        str(fixture_row.get("reference_audio"))
                        if fixture_row is not None and reference_audio is not None
                        else None
                    ),
                    "reference_audio_sha256": sha256_file(reference_audio),
                    "ok": call_ok,
                    "output_nonempty": output_nonempty,
                    "result": result_preview,
                    "violations": violations,
                }
            )
            if not call_ok:
                break
        steps["tools_call"] = {
            "ok": all_calls_ok and len(sample_evidence) == len(calls),
            "primary_field": primary_field,
            "output_nonempty": all(
                bool(sample.get("output_nonempty")) for sample in sample_evidence
            ) and len(sample_evidence) == len(calls),
            "text_nonempty": (
                all(bool(sample.get("output_nonempty")) for sample in sample_evidence)
                if primary_field == "text"
                else False
            ),
            "num_samples": len(sample_evidence),
            "expected_samples": len(calls),
            "samples": sample_evidence,
        }
        if not steps["tools_call"]["ok"]:
            raise RuntimeError(
                "tools/call step failed: "
                f"{json.dumps(sample_evidence[-1] if sample_evidence else payload, ensure_ascii=False)[:500]}"
            )

        shutdown_id = 3 + len(calls)
        if _send(proc, {"jsonrpc": "2.0", "id": shutdown_id, "method": "shutdown", "params": {}}, deadline):
            ok, payload = _read_response(proc.stdout.fileno(), stdout_buffer, shutdown_id, deadline, junk_tail)
        steps["shutdown"] = {"ok": ok}
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            server_code = proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            if server_code != 0 and not error:
                raise RuntimeError(f"server exited {server_code} after shutdown")
        except subprocess.TimeoutExpired:
            raise RuntimeError("server did not exit after shutdown within the deadline")
    except RuntimeError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - evidence must always be written
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    status = "passed" if error is None else "failed"
    payload = {
        "schema": SCHEMA,
        "status": status,
        "tool": args.tool,
        "audio": audio.name if audio is not None else None,
        "audio_sha256": sha256_file(audio),
        "fixture_gt_jsonl": fixture_gt_jsonl.name if fixture_gt_jsonl is not None else None,
        "fixture_gt_sha256": sha256_file(fixture_gt_jsonl),
        "initialize": steps["initialize"],
        "tools_list": steps["tools_list"],
        "tools_call": steps["tools_call"],
        "shutdown": steps["shutdown"],
        "server_command": portable_server_command(args.server_command),
        "server_stderr_tail": list(server_stderr)[-STDERR_TAIL_LINES:],
        "stdout_junk_count": len(junk_tail),
        "stdout_junk_tail": list(junk_tail)[-STDERR_TAIL_LINES:],
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "error": error,
    }
    try:
        produces.parent.mkdir(parents=True, exist_ok=True)
        produces.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as write_error:
        print(f"cannot write evidence {produces}: {write_error}", file=sys.stderr)
        return 1
    if stderr_handle is not None:
        stderr_handle.close()
    print(produces)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
