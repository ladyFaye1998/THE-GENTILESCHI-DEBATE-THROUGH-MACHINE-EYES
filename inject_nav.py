# inject_nav.py — universal global + section sub-nav + favicon injector (v5, "old money" style)
# Replaces previously injected navs automatically (same markers).
# Folders & logic unchanged; only the design is updated.

from __future__ import annotations
from pathlib import Path
from urllib.parse import quote
import argparse, os, re, sys

ROOT = Path(__file__).resolve().parent

# Canonical targets (relative to ROOT)
TARGETS = {
    "home":      Path("index.html"),
    "artemisia": Path("Artemisia/artemisia_dataset_overview.html"),
    "orazio":    Path("Orazio/orazio_dataset_overview.html"),
    "gender":    Path("Gender attribution/paintings_by_gender_overview.html"),
    "report":    Path("attribution_analysis_results/attribution_analysis_report.html"),
}

# Section-specific tabs
SECTION_TABS = {
    "artemisia": {
        "overview":     Path("Artemisia/artemisia_dataset_overview.html"),
        "unquestioned": Path("Artemisia/artemisia_unquestioned_gallery.html"),
        "questioned":   Path("Artemisia/artemisia_questioned_gallery.html"),
        "workshop":     Path("Artemisia/artemisia_workshop_gallery.html"),
    },
    "orazio": {
        "overview":     Path("Orazio/orazio_dataset_overview.html"),
        "unquestioned": Path("Orazio/orazio_unquestioned_gallery.html"),
        "questioned":   Path("Orazio/orazio_questioned_gallery.html"),
        "workshop":     Path("Orazio/orazio_workshop_gallery.html"),
    },
    "gender": {
        "overview": Path("Gender attribution/paintings_by_gender_overview.html"),
        "female":   Path("Gender attribution/female_paintings_gallery.html"),
        "male":     Path("Gender attribution/male_paintings_gallery.html"),
    },
}

# Markers so we can replace safely on re-run
NAV_MARK_START = "<!-- global+section-nav: injected -->"
NAV_MARK_END   = "<!-- /global+section-nav -->"
NAV_ID = 'id="global-nav-gentileschi"'

FAV_MARK_START = "<!-- favicon: injected -->"
FAV_MARK_END   = "<!-- /favicon -->"

def quote_path(posix_path: str) -> str:
    return "/".join(quote(seg) for seg in posix_path.split("/"))

def rel_href(from_file: Path, to_file: Path) -> str:
    rel = os.path.relpath(to_file, start=from_file.parent)
    return quote_path(Path(rel).as_posix())

def detect_top_section(html_file: Path) -> str:
    p = html_file.resolve().relative_to(ROOT).as_posix().lower()
    if p.startswith("artemisia/"): return "artemisia"
    if p.startswith("orazio/"): return "orazio"
    if p.startswith("gender attribution/"): return "gender"
    if p.startswith("attribution_analysis_results/"): return "report"
    if p.endswith("index.html"): return "home"
    name = html_file.name.lower()
    if "artemisia" in name: return "artemisia"
    if "orazio" in name: return "orazio"
    if any(w in name for w in ("gender","female","male")): return "gender"
    if "report" in name: return "report"
    return "home"

def detect_section_tab(section: str, html_file: Path) -> str | None:
    if section not in SECTION_TABS: return None
    name = html_file.name.lower()
    if "unquestioned" in name: return "unquestioned"
    if "questioned" in name:   return "questioned"
    if "workshop" in name:     return "workshop"
    if "female" in name:       return "female"
    if "male" in name:         return "male"
    if "overview" in name:     return "overview"
    return "overview"

# -------------- Build nav blocks (NEW aesthetic) --------------
def build_global_nav(html_file: Path, active: str) -> str:
    hrefs = {k: rel_href(html_file, ROOT / v) for k, v in TARGETS.items()}
    def li(key: str, label: str) -> str:
        cls = " gnav__item--active" if key == active else ""
        return f'<li class="gnav__item{cls}"><a href="{hrefs[key]}"><span>{label}</span></a></li>'
    return f"""
<style>
/* Old-money palette */
:root {{
  --ivory:#f6f1e9; --parch:#fbf9f5; --ink:#2b1e16; --soft:#6b5d54; --line:#d8cfc4;
  --deep:#423026; --accent:#b08d57; /* antique gold */
}}
/* Global bar */
.gnav{{position:sticky;top:0;z-index:1000;background:var(--parch);border-bottom:1px solid var(--line)}}
.gnav__wrap{{max-width:1200px;margin:0 auto;padding:10px 18px;display:flex;align-items:center;gap:18px}}
.gnav__brand a{{font:700 14px "Inter",system-ui,sans-serif;color:var(--ink);text-decoration:none;letter-spacing:.06em}}
.gnav__brand a::first-letter{{letter-spacing:.02em}}
.gnav__list{{list-style:none;display:flex;gap:18px;margin-left:8px}}
.gnav__item a{{display:block;padding:8px 2px;color:var(--ink);text-decoration:none}}
.gnav__item a span{{font:600 12px "Inter";text-transform:uppercase;letter-spacing:.11em;opacity:.88}}
.gnav__item a:hover span{{opacity:1}}
/* Active = refined gold underline */
.gnav__item--active a{{position:relative}}
.gnav__item--active a::after{{content:"";position:absolute;left:0;right:0;bottom:-8px;height:2px;background:var(--accent)}}
.gnav__spacer{{flex:1}}
.gnav__right a{{display:inline-block;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);font:600 12px "Inter";text-transform:uppercase;letter-spacing:.06em;text-decoration:none}}
.gnav__right a:hover{{border-color:var(--accent)}}
@media (max-width:720px){{.gnav__list{{flex-wrap:wrap}}}}

/* Section sub-nav */
.snav{{background:var(--ivory);border-bottom:1px solid var(--line)}}
.snav__wrap{{max-width:1200px;margin:0 auto;padding:10px 18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.snav__label{{font:700 12px "Inter";color:var(--soft);letter-spacing:.08em;text-transform:uppercase;margin-right:6px}}
.snav__tab{{display:inline-block;padding:6px 8px;border-radius:4px;color:var(--ink);text-decoration:none;border:1px solid transparent}}
.snav__tab:hover{{border-color:var(--line);background:#fff}}
.snav__tab--active{{border-color:var(--accent);background:#fff}}
.snav__tab--active::after{{content:"";display:block;height:2px;background:var(--accent);margin-top:4px}}
.snav__pill{{margin-left:6px;font:700 10px "Inter";border:1px solid var(--line);border-radius:5px;padding:2px 6px;background:#fff;color:var(--deep)}}
</style>
<nav {NAV_ID} class="gnav" aria-label="Global">
  <div class="gnav__wrap">
    <div class="gnav__brand"><a href="{hrefs['home']}">Gentileschi Project</a></div>
    <ul class="gnav__list">
      {li("home","Home")}
      {li("artemisia","Artemisia")}
      {li("orazio","Orazio")}
      {li("gender","Gender")}
      {li("report","ML Report")}
    </ul>
    <div class="gnav__spacer"></div>
    <div class="gnav__right"><a href="{hrefs['report']}">Open Analysis</a></div>
  </div>
</nav>
"""

def build_section_nav(html_file: Path, section: str, active_tab: str | None) -> str:
    if section not in SECTION_TABS: return ""
    tabs = SECTION_TABS[section]
    parts = ['<nav class="snav" aria-label="Section"><div class="snav__wrap">']
    label = "Artemisia" if section=="artemisia" else ("Orazio" if section=="orazio" else "By Gender")
    parts.append(f'<div class="snav__label">{label}</div>')
    for key, target in tabs.items():
        href = rel_href(html_file, ROOT / target)
        cls = " snav__tab--active" if key == active_tab else ""
        text = {"unquestioned":"Unquestioned","questioned":"Questioned","workshop":"Workshop",
                "female":"Female","male":"Male","overview":"Overview"}.get(key, key.capitalize())
        pill = ' <span class="snav__pill">Excluded from training</span>' if key=="workshop" else ""
        parts.append(f'<a class="snav__tab{cls}" href="{href}">{text}</a>{pill}')
    parts.append("</div></nav>")
    return "\n".join(parts)

def build_nav_block(html_file: Path) -> str:
    top = detect_top_section(html_file)
    active_tab = detect_section_tab(top, html_file) if top in SECTION_TABS else None
    return f"""{NAV_MARK_START}
{build_global_nav(html_file, active= top if top in {"artemisia","orazio","gender","report"} else "home")}
{build_section_nav(html_file, top, active_tab)}
{NAV_MARK_END}
"""

# ----------------- Favicon support (unchanged) -----------------
FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <rect x='2' y='2' width='60' height='60' rx='12' fill='#fbf9f5'/>
  <circle cx='32' cy='32' r='23' fill='#2b1e16'/>
  <text x='32' y='39' text-anchor='middle' font-size='28' font-family='Inter, Arial, sans-serif' font-weight='700' fill='#b08d57'>G</text>
</svg>
"""

def ensure_favicon_asset() -> Path:
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    svg_path = assets / "favicon.svg"
    if not svg_path.exists():
        svg_path.write_text(FAVICON_SVG, encoding="utf-8")
    return svg_path

def build_favicon_block(html_file: Path) -> str:
    svg_path = ensure_favicon_asset()
    rel_svg = rel_href(html_file, svg_path)
    return f"""{FAV_MARK_START}
<link rel="icon" type="image/svg+xml" href="{rel_svg}">
<link rel="shortcut icon" href="{rel_svg}">
{FAV_MARK_END}
"""

# ----------------- Injection helpers -----------------
def insert_or_replace_between(mark_start: str, mark_end: str, html: str, snippet: str) -> str|None:
    if mark_start in html and mark_end in html:
        return re.sub(re.escape(mark_start) + r".*?" + re.escape(mark_end),
                      snippet, html, flags=re.S)
    return None

def inject_into_head(html: str, snippet: str) -> str:
    m_open = re.search(r"(<head[^>]*>)", html, flags=re.I)
    if m_open: return html[:m_open.end()] + "\n" + snippet + html[m_open.end():]
    m_close = re.search(r"(</head>)", html, flags=re.I)
    if m_close: return html[:m_close.start()] + snippet + "\n" + html[m_close.start():]
    return snippet + html

def insert_or_replace_nav(html: str, snippet: str) -> str:
    if NAV_MARK_START in html and NAV_MARK_END in html:
        return re.sub(re.escape(NAV_MARK_START) + r".*?" + re.escape(NAV_MARK_END),
                      snippet, html, flags=re.S)
    html = re.sub(r"<nav[^>]*id=\"global-nav-gentileschi\"[^>]*>.*?</nav>", "", html, flags=re.S)
    m = re.search(r"(<body[^>]*>)", html, flags=re.I)
    return html[:m.end()] + "\n" + snippet + html[m.end():] if m else snippet + html

def strip_block(html: str, start: str, end: str, fallback_regex: str|None=None) -> str:
    if start in html and end in html:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), "", html, flags=re.S)
    if fallback_regex:
        return re.sub(fallback_regex, "", html, flags=re.S)
    return html

def should_touch(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".html"): return False
    if name.endswith(".bak"): return False
    if any(p.startswith(".") for p in path.parts): return False
    return True

def main():
    ap = argparse.ArgumentParser(description="Inject/remove refined global + section nav and favicon across the project.")
    ap.add_argument("--remove", action="store_true", help="Remove injected nav and favicon. Restores from .bak if present, else strips.")
    ap.add_argument("--favicon-only", action="store_true", help="Only (re)inject favicon; leave nav untouched.")
    args = ap.parse_args()

    changed = 0
    for html_file in ROOT.rglob("*.html"):
        if not should_touch(html_file): continue
        text = html_file.read_text(encoding="utf-8", errors="ignore")

        if args.remove:
            bak = html_file.with_suffix(html_file.suffix + ".bak")
            if bak.exists():
                html_file.write_text(bak.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                print(f"[restore] {html_file.relative_to(ROOT)}")
            else:
                t = strip_block(text, NAV_MARK_START, NAV_MARK_END, r"<nav[^>]*id=\"global-nav-gentileschi\"[^>]*>.*?</nav>")
                t = strip_block(t, FAV_MARK_START, FAV_MARK_END, r"<link[^>]+rel=[\"'](?:icon|shortcut icon)[\"'][^>]*>")
                html_file.write_text(t, encoding="utf-8")
                print(f"[strip]   {html_file.relative_to(ROOT)}")
            changed += 1
            continue

        fav_block = build_favicon_block(html_file)
        if not args.favicon_only:
            nav_block = build_nav_block(html_file)

        bak = html_file.with_suffix(html_file.suffix + ".bak")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")

        # favicon in <head>
        replaced = insert_or_replace_between(FAV_MARK_START, FAV_MARK_END, text, fav_block)
        text = inject_into_head(text, fav_block) if replaced is None else replaced

        if not args.favicon_only:
            text = insert_or_replace_nav(text, nav_block)

        html_file.write_text(text, encoding="utf-8")
        print(f"[inject {'favicon+nav' if not args.favicon_only else 'favicon'}] {html_file.relative_to(ROOT)}")
        changed += 1

    print(f"\nDone. Updated {changed} file(s). Backups: *.html.bak")

if __name__ == "__main__":
    sys.exit(main())
