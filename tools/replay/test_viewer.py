"""Real-browser test of tools/replay/index.html against tests/fixtures/trace_sample.

    make e2e-viewer     (= ../play-arc-agi-3/.venv/bin/python tools/replay/test_viewer.py)

Loads the page from file://, feeds the fixture through the folder input, and asserts on rendered
geometry and canvas pixels, not only DOM state. Needs a Playwright Chromium under ~/.cache/ms-playwright
(or PLAYWRIGHT_CHROMIUM=/path/to/chrome). Screenshots go to $VIEWER_SHOTS (default: ./viewer-shots).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
PAGE = (ROOT / "tools/replay/index.html").as_uri()
FIXTURE = ROOT / "tests/fixtures/trace_sample"
SHOTS = Path(os.environ.get("VIEWER_SHOTS", "viewer-shots"))
RESULTS: list[bool] = []

BLACK, GREY, RED, GREEN = [0, 0, 0], [170, 170, 170], [255, 65, 54], [46, 204, 64]
BLUE, YELLOW, PINK = [0, 116, 217], [255, 220, 0], [240, 18, 190]


def check(name: str, cond: object, detail: str = "") -> None:
    RESULTS.append(bool(cond))
    print(
        f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}",
        flush=True,
    )


def chromium() -> str:
    if os.environ.get("PLAYWRIGHT_CHROMIUM"):
        return os.environ["PLAYWRIGHT_CHROMIUM"]
    found = sorted(
        glob.glob(
            str(Path.home() / ".cache/ms-playwright/chromium-*/chrome-linux*/chrome")
        ),
        key=lambda p: int(p.split("chromium-")[1].split("/")[0]),
    )
    if not found:
        sys.exit(
            "no Chromium found: run `uvx playwright install chromium` or set PLAYWRIGHT_CHROMIUM"
        )
    return found[-1]


def main() -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium(), args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(PAGE)

        def px(x: int, y: int) -> list[int]:
            return page.evaluate(
                "([x,y]) => Array.from(document.getElementById('board').getContext('2d').getImageData(x,y,1,1).data).slice(0,3)",
                [x, y],
            )

        def label() -> str:
            return page.inner_text("#steplabel")

        print("loading")
        check("drop zone visible before load", page.locator("#drop").is_visible())
        page.set_input_files("#folder", str(FIXTURE))
        page.wait_for_selector("#games tbody tr")
        check("no page error while loading", not errors, str(errors))
        rows = page.locator("#games tbody tr")
        box = rows.first.bounding_box()
        check(
            "overview row renders with size",
            box and box["width"] > 100 and box["height"] > 20,
            str(box),
        )
        check("one game row", rows.count() == 1)
        txt = rows.first.inner_text()
        check(
            "levels 1 of 3, fallbacks 1, vs baseline 12/6 = 2.0x",
            "1 of 3" in txt and "2.0x" in txt,
            repr(txt),
        )
        check("controls text", "1-6" in txt, repr(txt))
        check(
            "tiles render",
            page.locator("#tiles .tile").count() == 4
            and "12" in page.inner_text("#tiles"),
        )
        page.screenshot(path=str(SHOTS / "overview.png"))

        print("replay")
        rows.first.click()
        page.wait_for_selector("#board", state="visible")
        b = page.locator("#board").bounding_box()
        check(
            "canvas non-zero and square",
            b and b["width"] > 200 and abs(b["width"] - b["height"]) < 1.5,
            str(b),
        )
        check("frame 0: grey block cell (11,11)", px(11, 11) == GREY, str(px(11, 11)))
        check("frame 0: red cell (41,41)", px(41, 41) == RED, str(px(41, 41)))
        check("frame 0: background (0,0)", px(0, 0) == BLACK)
        page.click("#b-next")
        check("diff step: (11,11) cleared", px(11, 11) == BLACK, str(px(11, 11)))
        check("diff step: (14,11) now grey", px(14, 11) == GREY, str(px(14, 11)))
        bb = page.locator("#bbox").bounding_box()
        check(
            "change bbox overlay has size",
            bb and bb["width"] > 10 and bb["height"] > 10,
            str(bb),
        )
        check("motion arrow drawn", page.locator("#ov .arrow").count() == 1)
        check("obs text shown for step 0", page.locator("#obstext").is_visible())
        check("model reply shown", "grey block" in page.inner_text("#replytext"))
        check(
            "plan highlights current move",
            page.locator("#p-plan .chip.cur").count() == 1,
        )
        page.click("#b-next")
        page.click("#b-next")
        check("click step: (41,41) now green", px(41, 41) == GREEN, str(px(41, 41)))
        mk = page.locator("#ov .click-marker").bounding_box()
        check("click marker has size", mk and mk["width"] > 5, str(mk))
        before = page.evaluate("document.getElementById('board').toDataURL()")
        page.click("#b-next")  # step 3: nothing changed
        same = page.evaluate("document.getElementById('board').toDataURL()")
        check("nothing-changed step leaves pixels equal", before == same)
        page.click("#b-next")  # step 4 fallback: bar of colour 7 at y=62
        after = page.evaluate("document.getElementById('board').toDataURL()")
        check("stepping next changes the pixels", after != same)
        check("fallback error shown", "no tool call" in page.inner_text("#errbox"))

        print("scrubber and timeline")
        page.evaluate(
            "() => { const s = document.getElementById('scrub'); s.value = 6; s.dispatchEvent(new Event('input', {bubbles:true})); }"
        )
        check("scrubber moves to step 5", label() == "step 5", label())
        check("keyframe: (0,0) blue", px(0, 0) == BLUE, str(px(0, 0)))
        check("keyframe: (22,22) yellow", px(22, 22) == YELLOW, str(px(22, 22)))
        check("keyframe: (51,6) pink", px(51, 6) == PINK, str(px(51, 6)))
        check("LEVEL CLEARED badge visible", page.locator("#cleared").is_visible())
        page.click('#bars .bar[data-frame="2"]')
        check("bar click changes step", label() == "step 1", label())
        check(
            "badge hidden on a normal step", not page.locator("#cleared").is_visible()
        )
        clear = page.locator("#bars .bar.clear")
        check(
            "exactly one level-clear bar, at step 5",
            clear.count() == 1 and clear.get_attribute("data-frame") == "6",
        )
        grey = page.locator("#bars .bar.same")
        check("nothing-changed bars grey (steps 3, 8, 11)", grey.count() == 3)
        h = page.evaluate(
            "[...document.querySelectorAll('#bars .bar')].map(b=>b.getBoundingClientRect().height)"
        )
        check(
            "bar heights: clear tallest, same shortest",
            h[5] == max(h) and h[3] == min(h) and min(h) > 0,
            str(h),
        )
        page.click("#b-last")
        check("last button", label() == "step 11", label())
        page.click("#b-first")
        check("first button", label() == "start")
        page.click("#b-play")
        page.wait_for_function(
            "document.getElementById('steplabel').textContent !== 'start'"
        )
        page.click("#b-play")
        check("play advances then pauses", page.inner_text("#b-play") == "Play")
        page.click('#bars .bar[data-frame="2"]')
        page.screenshot(path=str(SHOTS / "replay.png"), full_page=True)

        print("failures")
        page.click('.tab[data-view="failures"]')
        cards = page.locator(".fail-card")
        check(
            "two failure cards (fallback step 4, failed call step 8)",
            cards.count() == 2,
        )
        c0 = cards.first.bounding_box()
        check("failure card has size", c0 and c0["width"] > 200 and c0["height"] > 80)
        check(
            "fallback card shows the error", "no tool call" in cards.first.inner_text()
        )
        page.screenshot(path=str(SHOTS / "failures.png"), full_page=True)
        page.locator('.fail-card[data-step="4"] .jump').click()
        check(
            "jump opens the replay at that step",
            label() == "step 4" and page.locator("#board").is_visible(),
            label(),
        )
        check(
            "jumped board shows fallback bar (7 = orange)",
            px(3, 62) == [255, 133, 27],
            str(px(3, 62)),
        )

        print("malformed input")
        page.set_input_files(
            "#files",
            [
                {
                    "name": "bad.jsonl",
                    "mimeType": "text/plain",
                    "buffer": b'not json\n{"v":1}\n',
                }
            ],
        )
        page.wait_for_selector("#error", state="visible")
        check(
            "malformed jsonl shows an error naming the file and line",
            "bad.jsonl line 1" in page.inner_text("#error"),
        )
        page.set_input_files(
            "#files",
            [{"name": "run.json", "mimeType": "application/json", "buffer": b"{oops"}],
        )
        check(
            "malformed run.json shows an error", "run.json" in page.inner_text("#error")
        )
        page.set_input_files(
            "#files",
            [{"name": "readme.txt", "mimeType": "text/plain", "buffer": b"hi"}],
        )
        check(
            "unrelated files give an error",
            "No trace files" in page.inner_text("#error"),
        )
        page.click('.tab[data-view="overview"]')
        check(
            "page still works, previous run kept",
            page.locator("#games tbody tr").count() == 1,
        )
        page.set_input_files("#folder", str(FIXTURE))
        page.wait_for_selector("#games tbody tr")
        check("good load clears the error", not page.locator("#error").is_visible())
        check("no page errors at all", not errors, str(errors))
        browser.close()
    ok = all(RESULTS)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
