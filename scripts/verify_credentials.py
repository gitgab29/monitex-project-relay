"""Prove the two credentials actually work, before the demo depends on them.

    python scripts/verify_credentials.py            # check both
    python scripts/verify_credentials.py --gemini   # just Gemini
    python scripts/verify_credentials.py --smtp     # just Gmail (sends one real email)

Pasting a key into .env and seeing no error proves nothing: every credential path in this
project degrades quietly on purpose, so a wrong key looks exactly like a working one until
the demo. This makes a real call and a real send, and prints the specific thing that is
wrong when it fails rather than a stack trace.

Secrets are never printed -- only whether they are present and how long they are.
"""

from __future__ import annotations

import argparse
import sys
import time

from relay.config import load_settings


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str, fix: str = "") -> None:
    print(f"  [FAIL] {msg}")
    if fix:
        print(f"         -> {fix}")


def check_gemini(cfg) -> bool:
    print("\nGEMINI")
    if not cfg.gemini_api_key:
        bad("GEMINI_API_KEY is empty",
            "get one at https://aistudio.google.com/apikey and put it in .env")
        return False
    key = cfg.gemini_api_key
    ok(f"GEMINI_API_KEY present ({len(key)} chars, starts {key[:4]}...)")
    if not key.startswith("AIza"):
        bad("that does not look like a Google API key (they start with 'AIza')",
            "you may have copied a project id or an OAuth client id by mistake")
    ok(f"GEMINI_MODEL = {cfg.gemini_model}")

    try:
        from google import genai
    except ImportError:
        bad("google-genai is not installed", "pip install -e .")
        return False

    try:
        client = genai.Client(api_key=key)
        t0 = time.time()
        resp = client.models.generate_content(
            model=cfg.gemini_model,
            contents="Reply with exactly the word: pong",
        )
        dt = time.time() - t0
    except Exception as e:  # noqa: BLE001 - we want to explain, not propagate
        msg = str(e)
        bad(f"the call failed: {type(e).__name__}: {msg[:200]}")
        low = msg.lower()
        if "api key not valid" in low or "api_key_invalid" in low:
            bad("", "the key is wrong or was revoked -- make a new one at "
                    "https://aistudio.google.com/apikey")
        elif "429" in msg or "resource_exhausted" in low or "quota" in low:
            bad("", "rate limited. Check https://ai.google.dev/gemini-api/docs/rate-limits "
                    "and raise VISION_MIN_INTERVAL_S in .env")
        elif "404" in msg or "not found" in low:
            bad("", f"the model {cfg.gemini_model!r} is not available to this key. See "
                    "https://ai.google.dev/gemini-api/docs/models and set GEMINI_MODEL")
        elif "permission" in low or "403" in msg:
            bad("", "the key exists but is not enabled for the Gemini API -- regenerate it "
                    "from AI Studio, not from a Cloud console project")
        return False

    text = (getattr(resp, "text", "") or "").strip()
    ok(f"live call succeeded in {dt:.1f}s, model replied: {text[:40]!r}")
    if cfg.vision_min_interval_s:
        rpm = 60.0 / cfg.vision_min_interval_s
        ok(f"VISION_MIN_INTERVAL_S = {cfg.vision_min_interval_s} "
           f"-> at most {rpm:.0f} calls/min. Confirm your tier allows that at "
           f"https://ai.google.dev/gemini-api/docs/rate-limits")
    return True


def check_smtp(cfg) -> bool:
    print("\nGMAIL / SMTP")
    missing = [n for n, v in (("SMTP_USER", cfg.smtp_user),
                              ("SMTP_APP_PASSWORD", cfg.smtp_app_password),
                              ("ALERT_TO", cfg.alert_to)) if not v]
    if missing:
        bad(f"empty in .env: {', '.join(missing)}",
            "app password: https://myaccount.google.com/apppasswords "
            "(needs 2-Step Verification on first)")
        return False

    pw = cfg.smtp_app_password
    ok(f"SMTP_USER = {cfg.smtp_user}")
    ok(f"ALERT_TO  = {cfg.alert_to}")
    ok(f"SMTP_APP_PASSWORD present ({len(pw)} chars)")
    if " " in pw:
        bad("the app password contains spaces",
            "Google displays it as 4 groups of 4 -- paste it as 16 characters, no spaces")
        return False
    if len(pw) != 16:
        bad(f"app passwords are 16 characters, this is {len(pw)}",
            "you may have pasted your account password instead")
    ok(f"server = {cfg.smtp_host}:{cfg.smtp_port}")

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "[relay] credential check"
    msg["From"] = cfg.smtp_user
    msg["To"] = cfg.alert_to
    msg.set_content("If you are reading this, Project Relay can send mail. "
                    "Sent by scripts/verify_credentials.py.")

    try:
        t0 = time.time()
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20) as s:
            s.starttls()
            s.login(cfg.smtp_user, cfg.smtp_app_password)
            s.send_message(msg)
        dt = time.time() - t0
    except smtplib.SMTPAuthenticationError as e:
        bad(f"authentication rejected: {e.smtp_code} {e.smtp_error!r:.120}")
        bad("", "535 here almost always means a normal account password was used, or "
                "2-Step Verification is off. Make an APP password at "
                "https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:  # noqa: BLE001
        bad(f"send failed: {type(e).__name__}: {str(e)[:200]}")
        return False

    ok(f"sent in {dt:.1f}s -- check the inbox of {cfg.alert_to}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemini", action="store_true", help="check Gemini only")
    ap.add_argument("--smtp", action="store_true", help="check Gmail only (sends an email)")
    ap.add_argument("--env", default=".env", help="path to the env file")
    args = ap.parse_args()

    both = not (args.gemini or args.smtp)
    cfg = load_settings(args.env)
    print(f"loaded {args.env}")

    results = []
    if both or args.gemini:
        results.append(("gemini", check_gemini(cfg)))
    if both or args.smtp:
        results.append(("smtp", check_smtp(cfg)))

    print("\n" + "-" * 52)
    for name, good in results:
        print(f"  {name:8} {'PASS' if good else 'FAIL'}")
    return 0 if all(g for _, g in results) else 1


if __name__ == "__main__":
    sys.exit(main())
