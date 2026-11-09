"""Reword the X post's text into a headline in the clip-page style."""
import nowin  # noqa: F401  (no console windows for child processes)
import json
import os
import re
import subprocess
import time

import requests

PROMPT = """Reword this X post into a headline for a TikTok streamer clip page.

X post (by @{author}): "{tweet}"{said}

Return ONLY JSON: {{"relevant": true/false, "adult": true/false, "kind": "...", "funny": 0-10, "headline": "...", "emoji": "..."}}

Rules for the headline:
- Same story as the post, but mix up the words so it isn't copied word for word.
- One sentence, 8-22 words, third person, says who does what.
- Words in quotes are what someone SAID (a joke, a story, a line). Never turn them into something that
  happened: "he said Reggie kept going" is DDG joking, not Reggie doing anything.
{lead}- Use exactly the people named in the post, spelled the same way. Never add, drop or swap a person.
- Keep the key details exactly as the post has them: song and track names, artists, brands, places,
  numbers and money. If the post explains a name (e.g. a remix of "Yale" titled "Jail"), keep the
  explanation, use the better-known name ("Yale") rather than only the nickname.
- No hashtags, no @, no links. Don't add anything the post doesn't say.
"kind": what the clip is: "funny" (someone being hilarious, a funny reaction, a roast, an awkward or
embarrassing moment, people dying of laughter), "drama" (beef, arguments, fights), "hype" (big performance,
flex, surprise appearance), "news" (announcements, updates, rumours, schedules), "promo" (ads, links, merch).
"funny": 0-10, how funny the clip sounds from the post. 8+ = people will be crying laughing; 5 = mildly
amusing; 0-2 = not a joke at all (news, updates, promos). Laughing emojis alone don't make it funny.
"adult": true if the post is about porn, OnlyFans, sex work, nudity or other adult content.
"emoji": one that fits, from 😭 🤣 😳 💀 😂 😱 🔥
"relevant": true only if the post is about {who}. False if it's about something else that shares the
name (e.g. neon lights, a K-pop group or song called Neon, a different person called Lacy)."""


def _gpu_uuid(name):
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        gname, uuid = [x.strip() for x in line.split(",")]
        if name in gname:
            return uuid
    return None


def ollama_url(cfg):
    """The bot's own Ollama, locked to one GPU (the GTX 1660), on its own port.
    Your normal Ollama on port 11434 is left alone. Falls back to it if the GPU isn't found."""
    gpu = cfg.get("ollama_gpu")
    uuid = _gpu_uuid(gpu) if gpu else None
    if not uuid:
        return "http://localhost:11434"
    url = f"http://127.0.0.1:{cfg.get('ollama_port', 11435)}"
    try:
        requests.get(f"{url}/api/version", timeout=2)
        return url
    except requests.RequestException:
        pass
    env = {**os.environ, "OLLAMA_HOST": url.removeprefix("http://"), "CUDA_VISIBLE_DEVICES": uuid,
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OLLAMA_VULKAN": "0"}  # Vulkan would grab the other card
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    for _ in range(30):
        time.sleep(1)
        try:
            requests.get(f"{url}/api/version", timeout=2)
            return url
        except requests.RequestException:
            pass
    return "http://localhost:11434"


def _serve(url, env):
    try:
        requests.get(f"{url}/api/version", timeout=2)
        return True
    except requests.RequestException:
        pass
    subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_HOST": url.removeprefix("http://"), **env},
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    for _ in range(30):
        time.sleep(1)
        try:
            requests.get(f"{url}/api/version", timeout=2)
            return True
        except requests.RequestException:
            pass
    return False


def cpu_ollama_url(cfg):
    """A CPU-only Ollama on its own port, for when the GPUs are busy."""
    url = f"http://127.0.0.1:{cfg.get('ollama_cpu_port', 11436)}"
    _serve(url, {"CUDA_VISIBLE_DEVICES": "-1", "OLLAMA_VULKAN": "0", "HIP_VISIBLE_DEVICES": "-1"})
    return url


def _unload(cfg):
    """Drop the model from the 1660's memory so a render can use it."""
    url = f"http://127.0.0.1:{cfg.get('ollama_port', 11435)}"
    try:
        requests.post(f"{url}/api/generate", json={"model": cfg["ollama_model"], "keep_alive": 0}, timeout=10)
    except requests.RequestException:
        pass


def _ask(cfg, prompt):
    if os.environ.get("ANTHROPIC_API_KEY") and cfg.get("llm") == "anthropic":
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=cfg["anthropic_model"], max_tokens=400, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    body = {"model": cfg["ollama_model"], "stream": False, "format": "json", "think": False, "keep_alive": "30m",
            "options": {"temperature": 0.7}, "messages": [{"role": "user", "content": prompt}]}
    import layout
    if layout.blender_running():
        # Blender is rendering: free the 1660 and think on the CPU instead.
        _unload(cfg)
        r = requests.post(f"{cpu_ollama_url(cfg)}/api/chat", timeout=300, json={**body, "keep_alive": "5m"})
        r.raise_for_status()
        return r.json()["message"]["content"]
    url = ollama_url(cfg)
    try:
        # First call after a while loads the model onto the 1660, which can take a minute or two.
        r = requests.post(f"{url}/api/chat", timeout=150, json=body)
    except requests.RequestException:
        # The 1660 is stuck or busy: think on the CPU instead (your normal Ollama may not be running).
        r = requests.post(f"{cpu_ollama_url(cfg)}/api/chat", timeout=300, json={**body, "keep_alive": "5m"})
    r.raise_for_status()
    return r.json()["message"]["content"]


def clean_tweet(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[@#]\w+", lambda m: m.group(0)[1:], text)
    return " ".join(text.split())


CENSOR = [
    (r"\b(?:used|uses|using|dropped|drops|dropping|says|said|saying) (?:the |an? )?n[- ]?words?\b", "said Ni**a"),
    (r"\b(?:the |an? )?n[- ]?words?\b", "Ni**a"),
    (r"\bnigg(?:a|er)s?\b", "Ni**a"),
    (r"\bfuck", "F**k"),
    (r"\bshit", "Sh*t"),
    (r"\bbitch", "B***h"),
]


def censor(text):
    """Owner's style: slurs and swears written with asterisks, e.g. Ni**a."""
    for pat, rep in CENSOR:
        text = re.sub(pat, rep, text, flags=re.I)
    return text


# Adult-content posts get skipped (TikTok takes them down / limits the account).
ADULT = re.compile(r"🌽|\bonly ?fans\b|ø ?f\b|\bporn|\bp\*rn|\bnsfw\b|\bnudes?\b|\bxxx\b|\bstripper|\bsex\b", re.I)


# Korean / Chinese / Japanese text: not our US streamers, and the font can't draw it anyway.
FOREIGN = re.compile("[぀-ヿ㐀-鿿가-힯]")


# "Neon" / "Lacy" are normal words (a song, a K-pop group, a dress). When that's the only name in the
# post, it must also talk about streaming, or name another streamer, to count.
STREAM_CONTEXT = re.compile(r"\bstream|\bkick\b|\btwitch|\bchat\b|\blive\b|\birl\b|\bkai\b|cenat|adin|"
                            r"ishowspeed|\bspeed\b|plaqueboymax|\bddg\b|jasontheween|n3on|fortnite|\bclip|"
                            r"\bfans?\b|\bcasino|\bdonat|\breact|\bcrash(?:es|ed)? out|\bviewers?", re.I)


def unmistakable(cfg, text, found):
    """False when the only creator match is an everyday word with nothing about streaming around it."""
    for k in found:
        names = [n.strip('"').lower() for n in [k, *cfg["creators"][k]["names"]] if n.strip('"').lower() not in COMMON_WORDS]
        if any(re.search(rf"\b{re.escape(n)}\b", text, re.I) for n in names):
            return True                  # written unambiguously (n3on, plaqueboymax, ddg ...)
        # Clip posts start with the name ("Lacy tries Japanese matcha..."); IRL trips have no stream words.
        if any(re.match(rf"\s*{n[0].upper()}(?:{n[1:]}|{n[1:].upper()})\b", text)
               for n in [k, *cfg["creators"][k]["names"]] if n.lower() in COMMON_WORDS):
            return True
    return bool(STREAM_CONTEXT.search(text))


def leads(cfg, headline):
    """Creators the clip is actually ABOUT: named in the headline's first 3 words.
    "JasonTheWeen reacts to ... PlaqueBoyMax" -> [jasontheween] only (owner 2026-11-06: a campaign gets
    only clips of its own creator, not someone reacting to or talking about him)."""
    words = re.findall(r"\S+", headline or "")[:3]
    out = []
    for w in words:
        for k in creators_named(cfg, w.strip(",.!?:'’s")):
            if k not in out:
                out.append(k)
    return out


def creators_named(cfg, text):
    """Creators whose name is actually written in the text (the LLM over-tags)."""
    def named(n):
        n = n.strip('"').lower()
        if n == "jason":        # clip pages write just "Jason" (owner 2026-11-07); not Jason Derulo / Kelce / Momoa
            return re.search(r"\bJason\b(?![\s’']+(?:[A-Z][a-z]|[A-Z]{2}))", text) or re.search(r"\bJason[’']s\b", text)
        if n in COMMON_WORDS:   # "neon sign", "lacy dress": only counts written as a name (Neon, LACY)
            return re.search(rf"\b{n[0].upper()}(?:{n[1:]}|{n[1:].upper()})\b", text)
        return re.search(rf"\b{re.escape(n)}\b", text, re.I)
    return [k for k, c in cfg["creators"].items() if any(named(n) for n in [k, *c["names"]])]


COMMON_WORDS = {"neon", "lacy", "dub", "jason"}


POLITICS = re.compile(r"\b(obama|trump|biden|kamala|harris|aipac|election|senator|senate|congress|democrats?|republicans?|"
                      r"campaign rally|elsayed|governor)\b", re.I)
# Song snippets / previews get muted or set to "Only me" by TikTok for copyright.
MUSIC = re.compile(r"\bsnippets?\b|\bunreleased\b|\b(new|upcoming) (song|track|single|music)\b|\b(previews?|drops?|dropped|"
                   r"plays|played|teases?|teased) (a |his |her |their |the )?(new )?(song|track|single|music)\b|"
                   r"\bmusic video\b|\bleaked (song|track)\b", re.I)


def meme_list(tweet):
    """Joke list posts ("Paragraphs and black trucks from Big H, AIPAC, Nike, Redbull, ... after one stream")
    are captions over unrelated footage, not a clip of the moment."""
    return tweet.count(",") >= 5


QUOTE = re.compile(r'[“"]([^“”"]{12,})[”"]')


def split_quote(tweet):
    """(story, quote): the post's own description with the quoted speech taken out, so the headline is
    written from what happened, not from a line someone said. Only when enough story is left."""
    quotes = QUOTE.findall(tweet)
    story = re.sub(r"\s+", " ", QUOTE.sub(" ", tweet)).strip(" ,:-")
    if not quotes or len(re.findall(r"[A-Za-z']+", story)) < 6:
        return tweet, ""
    return story, " / ".join(q.strip(" .") for q in quotes)


NOT_NAMES = set("""when the a an this that after before how why what pov breaking watch imagine bro chat
fans fan streamer streamers someone everyone nobody paragraphs""".split())


def subject(story):
    """Who the post is about: its first word when that's a name ("Tylil GOES OFF on Reggie..." -> Tylil)."""
    w = re.findall(r"[A-Za-z0-9][A-Za-z0-9']*", story)
    if not w or w[0].lower() in NOT_NAMES or not (w[0][0].isupper() or w[0][0].isdigit()):
        return None
    return w[0].split("'")[0]


def write(cfg, cand):
    tweet = clean_tweet(cand["text"])
    # Needs an actual story ("DDG FUNNY 🤣🤣" isn't one): at least 6 real words besides the names.
    words = [w for w in re.findall(r"[A-Za-z']+", tweet)
             if not creators_named(cfg, w) and w.lower() not in {"funny", "lol", "lmao", "bro", "crazy", "wild", "clip"}]
    if len(words) < 6:
        return {"relevant": False, "creators": [], "headline": "", "emoji": ""}
    # Tags come only from the post itself: the LLM sometimes swaps in a creator who isn't there.
    found = creators_named(cfg, tweet)
    if found and not unmistakable(cfg, tweet, found):
        found = []
    if not found or ADULT.search(cand["text"]) or FOREIGN.search(cand["text"]) or POLITICS.search(tweet) or meme_list(tweet) or MUSIC.search(tweet):
        return {"relevant": False, "creators": found, "headline": "", "emoji": ""}
    if False:
        return {"relevant": False, "creators": [], "headline": "", "emoji": ""}
    who = " / ".join(cfg["creators"][k].get("about", k) for k in found)
    story, quote = split_quote(tweet)
    said = f'\nQuote said in the clip (words, not events): "{quote}"' if quote else ""
    # The 8B model swaps the post's main person for a creator it knows ("Tylil goes off" -> "DDG roasts").
    lead_name = subject(story)
    # "Marlon and JasonTheWeen couldn't..." -> lead with the creator we clip, not his friend.
    opening = " ".join(re.findall(r"\S+", story)[:4])
    if creators_named(cfg, opening) and not creators_named(cfg, lead_name or ""):
        lead_name = next((w.strip(",.'’s") for w in opening.split() if creators_named(cfg, w)), lead_name)
    lead =(f"- The post is about {lead_name}: start the headline with {lead_name}, doing what the post "
            f"says {lead_name} does.\n") if lead_name else ""
    for _ in range(3):
        raw = _ask(cfg, PROMPT.format(author=cand["author"], tweet=story, said=said, who=who, lead=lead))
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        headline = censor(re.sub(r"[#@]\w+", "", data.get("headline", "")).strip().rstrip("."))
        # The headline must name one of OUR creators (never just Marlon/Emily/Reggie).
        if (headline and creators_named(cfg, headline) and set(creators_named(cfg, headline)) <= set(found)
                and (not lead_name or lead_name.lower() in headline.lower())):
            break
    else:
        return {"relevant": False, "creators": found, "headline": "", "emoji": ""}
    emoji = (data.get("emoji") or "😭").strip()[:1]
    try:
        funny = float(data.get("funny") or 0)
    except (TypeError, ValueError):
        funny = 0
    kind = str(data.get("kind") or "").lower()
    return {"relevant": bool(data.get("relevant")) and not data.get("adult")
                        and bool(found) and bool(headline),
            "creators": found, "headline": headline, "emoji": emoji, "funny": funny, "kind": kind}


def description(cfg, info):
    tags = [t for k in info["creators"] for t in cfg["creators"][k]["tags"]] + cfg["extra_tags"]
    seen = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    text = info["headline"][0].upper() + info["headline"][1:]
    return f"{text} {info['emoji']} {' '.join(seen)}"


STORY_PROMPT = """A clip page is about to post a new clip. Is it the SAME MOMENT as one it already posted?
Same moment = the same thing happening (e.g. two posts both about Cuffem being introduced as a judge).
NOT the same = a different thing that happened on the same stream or event.

New clip: "{new}"

Already posted:
{posted}

Return ONLY JSON: {{"same_as": <number of the matching posted clip, or 0 if none>}}"""


def same_story(cfg, new_text, posted_texts):
    """Index into posted_texts of a clip showing the same moment, or None."""
    if not posted_texts:
        return None
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(posted_texts))
    try:
        raw = _ask(cfg, STORY_PROMPT.format(new=new_text, posted=listing))
        n = int(json.loads(raw[raw.find("{"): raw.rfind("}") + 1]).get("same_as") or 0)
    except Exception:
        return None
    if not 1 <= n <= len(posted_texts):
        return None
    # The 8B model often picks some post off a long list: double-check that one pair on its own,
    # and they have to share a real word (Halloween costume vs earphones at a party = different).
    old = posted_texts[n - 1]
    if not _shared_words(new_text, old):
        return None
    try:
        raw = _ask(cfg, PAIR_PROMPT.format(a=new_text, b=old))
        same = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]).get("same")
    except Exception:
        return None
    return n - 1 if same is True else None


PAIR_PROMPT = """Do these two clip-page posts show the SAME MOMENT (the same thing happening on camera)?
Two different things from the same stream, party or trip are NOT the same moment.

A: "{a}"
B: "{b}"

Return ONLY JSON: {{"same": true/false}}"""

_FILLER = set("""the and his her their with after while about that this what when then from into over just they them
was were has had have been being said says saying got gets made makes make like after before during stream
clip live chat says said told tells because couldn't can't didn't doesn't wasn't don't really even still""".split())


def _shared_words(a, b):
    """Content words (4+ letters, not filler) the two posts have in common, besides names they share anyway."""
    words = lambda t: {w.lower() for w in re.findall(r"[A-Za-z']{4,}", t)} - _FILLER
    common = words(a) & words(b)
    names = {"ddg", "plaqueboymax", "jasontheween", "lacy", "n3on", "darla", "stableronaldo", "adapt"}
    return common - names
