"""Post a video to TikTok through TikTok Studio in a browser that stays logged in.

Log in once with `python login.py`; the bot reuses that browser profile.
"""
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
PROFILE = ROOT / "browser_profile"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/156.0.0.0 Safari/537.36 Edg/156.0.0.0")
UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload?from=webapp"


def _context(p, headless):
    return p.chromium.launch_persistent_context(
        str(PROFILE), channel="msedge", headless=headless, viewport={"width": 1280, "height": 900},
        # Hidden browser says "HeadlessChrome" in its user agent; send the normal Edge one instead.
        user_agent=UA if headless else None,
        args=["--disable-blink-features=AutomationControlled"])


def login(url="https://www.tiktok.com/login", site="TikTok"):
    with sync_playwright() as p:
        ctx = _context(p, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url)
        print(f"Log in to {site} in the window, then close the window.")
        while ctx.pages:
            time.sleep(1)
        ctx.close()


def _click_if(page, text, timeout=1500):
    btn = page.get_by_role("button", name=text)
    try:
        btn.first.click(timeout=timeout)
        return True
    except Exception:
        return False


def _dismiss_popups(page):
    """Close TikTok's one-off popups and tips without changing any settings."""
    for _ in range(4):
        clicked = False
        # Settings prompts ("Turn on automatic content checks?"): say Cancel.
        prompt = page.locator("[role=dialog], .TUXModal, [class*='modal'], [class*='Modal']").filter(
            has_text=re.compile(r"content check|Turn on", re.I))
        if prompt.count():
            clicked = _click_if(prompt.first, "Cancel", 1000)
        # Tips and tours ("New editing features added"): any visible Got it / Not now / Skip.
        for name in ("Got it", "Not now", "Skip", "Dismiss"):
            btn = page.get_by_role("button", name=re.compile(rf"^\s*{name}\s*$", re.I))
            for i in range(btn.count()):
                try:
                    if btn.nth(i).is_visible():
                        btn.nth(i).click(timeout=1000)
                        clicked = True
                except Exception:
                    pass
        if not clicked:
            return
        page.wait_for_timeout(800)


def _type_description(page, editor, description):
    editor.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")
    for word in description.split(" "):
        page.keyboard.type(word, delay=25)
        if word.startswith("#"):
            page.wait_for_timeout(1200)   # let the hashtag popup show, then move on without picking
            page.keyboard.press("Escape")
        page.keyboard.type(" ")
    page.wait_for_timeout(1000)


def post(video: Path, description: str, headless=False, log=print):
    with sync_playwright() as p:
        ctx = _context(p, headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(UPLOAD_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
            if "login" in page.url:
                raise RuntimeError("TikTok is logged out: run `python login.py` once")

            page.locator("input[type=file]").first.set_input_files(str(video))

            editor = page.locator("div[contenteditable='true']").first
            editor.wait_for(timeout=120_000)
            page.wait_for_timeout(3000)
            _dismiss_popups(page)
            # Type the description, and check it really went in (a popup can swallow the keys).
            first_words = description.split(" #")[0][:25]
            for attempt in range(3):
                _type_description(page, editor, description)
                if first_words in editor.inner_text():
                    break
                _dismiss_popups(page)
            else:
                raise RuntimeError("couldn't type the description")

            # Wait for upload to finish: the Post button turns enabled.
            post_btn = page.locator("button[data-e2e='post_video_button']")
            deadline = time.time() + 600
            while time.time() < deadline:
                if post_btn.count() and post_btn.first.get_attribute("aria-disabled") != "true" \
                        and post_btn.first.is_enabled():
                    break
                page.wait_for_timeout(2000)
            else:
                raise RuntimeError("upload never finished")
            page.wait_for_timeout(2000)
            _dismiss_popups(page)
            post_btn.first.click()

            # TikTok sometimes asks "Continue to post?" while its content check runs.
            deadline = time.time() + 180
            while time.time() < deadline:
                if "tiktokstudio/content" in page.url:
                    log("posted")
                    return True
                if page.get_by_text("Your video has been uploaded").count() or \
                        page.get_by_text("Video published").count():
                    log("posted")
                    return True
                _click_if(page, "Post now")
                page.wait_for_timeout(2000)
            page.screenshot(path=str(ROOT / "logs" / f"post_fail_{int(time.time())}.png"))
            raise RuntimeError("never saw the posted confirmation")
        finally:
            ctx.close()
