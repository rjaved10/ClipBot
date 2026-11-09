"""Submit our posted TikToks to the matching clipping.net campaign bounty (PlaqueBoyMax -> pbm, JasonTheWeen -> jason).

Only clips that carry the campaign's own watermark (our name badge) are submitted: every campaign
says clips without the provided watermark don't get paid. Runs in the same hidden Edge profile as the
TikTok poster, after a post, so the two never use the browser at the same time.
"""
import json
import re
import time
from datetime import datetime, timedelta

from playwright.sync_api import sync_playwright

import poster


def bounty_for(cfg, badge, leads=(), rising=False):
    """The bounty a clip goes to: its badge is a campaign's watermark, or (campaigns with
    "badge": false, e.g. Lacy) the campaign's creator leads the headline.
    Rising Star stream clips go to that night's "Rising Star <Mon> <day>" bounty (owner 2026-11-07)."""
    c = cfg.get("clipping", {})
    if not c.get("enabled"):
        return None
    camps = c.get("campaigns", {})
    if badge and badge.stem in camps:
        camp = camps[badge.stem]
        if rising and camp.get("rising_star"):
            d = datetime.now() - timedelta(hours=5)     # a stream running past midnight keeps its start day
            return f"{camp['slug']}|Rising Star {d:%b} {d.day}"
        return f"{camp['slug']}|{camp['bounty']}"
    for k in leads:
        camp = camps.get(k)
        if camp and camp.get("badge") is False and camp.get("slug"):
            return f"{camp['slug']}|{camp['bounty']}"
    return None


def _studio_items(page):
    """Our recent TikToks from TikTok Studio: item_id, desc, create_time, visibility."""
    items = []

    def grab(resp):
        if "creator/manage/item_list" in resp.url:
            try:
                items.extend(json.loads(resp.text()).get("item_list") or [])
            except Exception:
                pass
    page.on("response", grab)
    page.goto("https://www.tiktok.com/tiktokstudio/content", timeout=60000)
    page.wait_for_timeout(8000)
    page.remove_listener("response", grab)
    if not items:
        raise RuntimeError("couldn't read the TikTok Studio post list")
    return items


def _key(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())[:40]


def _submit(page, slug, bounty, url):
    page.goto(f"https://clipping.net/dashboard/campaigns/{slug}", timeout=60000)
    page.wait_for_timeout(4000)
    page.get_by_role("button", name="Upload Clip").first.click()
    page.wait_for_timeout(1500)
    page.locator("[role=dialog] textarea").fill(url)
    page.wait_for_timeout(2500)
    page.locator("[role=dialog]").get_by_role("combobox").first.click()
    page.wait_for_timeout(1000)
    opts = [o.inner_text().strip() for o in page.get_by_role("option").all()]
    if bounty not in opts and bounty.startswith("Rising Star"):
        # That night's Rising Star bounty isn't up (yet / any more): newest Rising Star one, else the main one.
        rs = [o for o in opts if o.startswith("Rising Star")]
        bounty = rs[-1] if rs else next((o for o in opts if not o.startswith("Rising Star")), bounty)
    page.get_by_role("option", name=bounty, exact=True).click()
    page.wait_for_timeout(1000)
    up = page.get_by_role("button", name=re.compile(r"Upload 1 Clip"))
    if not up.is_enabled():
        raise RuntimeError("upload button stayed disabled: " + page.locator("[role=dialog]").inner_text()[-200:])
    up.click()
    page.wait_for_timeout(5000)
    dlg = page.locator("[role=dialog]")
    if dlg.count() and dlg.first.is_visible():
        raise RuntimeError("form still open after upload: " + dlg.first.inner_text()[-200:])


def run(cfg, con, log):
    """Submit posted clips that are waiting for their clipping.net submission."""
    c = cfg.get("clipping", {})
    if not c.get("enabled"):
        return
    wait = datetime.now() - timedelta(minutes=c.get("delay_minutes", 3))
    rows = con.execute("select media_key, description, posted, bounty, clip_tries from clips where status='posted' "
                       "and bounty is not null and clip_sub is null and posted < ?", [wait.isoformat()]).fetchall()
    # The same clip on YouTube Shorts goes to the same campaign bounty (owner 2026-11-06).
    yt_rows = con.execute("select media_key, yt, bounty, yt_sub_tries, description from clips where status='posted' "
                          "and bounty is not null and yt like 'http%' and yt_sub is null").fetchall()
    # Clips made before the strict rule carry a loose bounty ("Lacy reacts to ... PlaqueBoyMax" -> pbm):
    # only submit when the campaign's creator leads the headline.
    import headline
    slug_owner = {v["slug"]: k for k, v in c.get("campaigns", {}).items()}
    keep = []
    for key, url, bounty, tries, desc in yt_rows:
        if slug_owner.get(bounty.split("|")[0]) in headline.leads(cfg, (desc or "").split(" #")[0]):
            keep.append((key, url, bounty, tries))
        else:
            con.execute("update clips set yt_sub='skip: not that campaign creator''s clip' where media_key=?", [key])
    con.commit()
    yt_rows = keep
    if not rows and not yt_rows:
        return
    with sync_playwright() as p:
        ctx = poster._context(p, cfg["browser_headless"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for key, url, bounty, tries in yt_rows:
                slug, name = bounty.split("|")
                tries = (tries or 0) + 1
                try:
                    _submit(page, slug, name, url)
                    done = f"submitted {name} {url}"
                    log(f"clipping.net: submitted YouTube Short to {name}: {url}")
                except Exception as e:
                    done = f"error: {str(e)[:200]}" if tries >= 3 else None
                    log(f"clipping.net: YouTube submit failed for {key}: {str(e)[:200]}")
                con.execute("update clips set yt_sub=?, yt_sub_tries=? where media_key=?", [done, tries, key])
                con.commit()
            if not rows:
                return
            items = _studio_items(page)
            for key, desc, posted, bounty, tries in rows:
                slug, name = bounty.split("|")
                since = time.mktime(datetime.fromisoformat(posted).timetuple()) - 900
                item = next((i for i in items if _key(i.get("desc")) == _key(desc)
                             and int(i.get("create_time") or 0) >= since), None)
                tries = (tries or 0) + 1
                if not item:
                    done = "not found on TikTok" if tries >= 6 else None
                    log(f"clipping.net: {key} not in TikTok Studio yet (try {tries})")
                elif item.get("visibility") != 1:
                    done = "skip: TikTok made it Only me"
                    log(f"clipping.net: not submitting {key}, TikTok set it to Only me")
                else:
                    url = f"https://www.tiktok.com/@{c['tiktok_handle']}/video/{item['item_id']}"
                    try:
                        _submit(page, slug, name, url)
                        done = f"submitted {name} {url}"
                        log(f"clipping.net: submitted to {name}: {url}")
                    except Exception as e:
                        done = f"error: {str(e)[:200]}" if tries >= 3 else None
                        log(f"clipping.net: submit failed for {key}: {str(e)[:200]}")
                con.execute("update clips set clip_sub=?, clip_tries=? where media_key=?", [done, tries, key])
                con.commit()
        finally:
            ctx.close()
