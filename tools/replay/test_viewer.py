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
FIXTURE_B = ROOT / "tests/fixtures/trace_sample_b"
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

        print("compare")
        page.set_viewport_size({"width": 1440, "height": 1400})
        page.click('.tab[data-view="compare"]')
        check(
            "one run: explanation shown",
            "Only one run is loaded" in page.inner_text("#c-msg"),
        )
        check(
            "one run: no canvases rendered",
            page.locator("#cv" + "A").count() == 0
            and not page.locator("#c-side").is_visible(),
        )
        check(
            "one run: table has one row",
            page.locator("#cmp-table tbody tr").count() == 1,
        )
        page.screenshot(path=str(SHOTS / "compare-one-run.png"), full_page=True)
        page.set_input_files("#folder-b", str(FIXTURE_B))
        page.wait_for_selector("#cvA", state="visible")
        check(
            "add-a-run names both runs",
            "trace_sample_b" in page.inner_text("#runnames")
            and "trace_sample" in page.inner_text("#runnames"),
            page.inner_text("#runnames"),
        )
        check("two table rows", page.locator("#cmp-table tbody tr").count() == 2)
        # A: 12 actions, 1 level, 1 fallback (8.3%), 4 calls... B: 6 actions, 1 level, 0 fallbacks
        best = page.evaluate(
            "() => [...document.querySelectorAll('#cmp-table td.best')].map(td => td.closest('tr').dataset.run + ':' + td.dataset.col)"
        )
        check(
            "best cells: B has fewer actions, no fallbacks, fewer tokens and seconds; levels tie",
            sorted(best) == ["B:actions", "B:fb", "B:sec", "B:tok"],
            str(best),
        )
        bb = page.locator("#cmp-table td.best").first.bounding_box()
        check(
            "best cell visible with size",
            bb and bb["width"] > 30 and bb["height"] > 15,
            str(bb),
        )
        check(
            "settings read from run.json",
            "timeout_s 180" in page.locator('#cmp-table tr[data-run="B"]').inner_text(),
        )

        def cpx(side: str, x: int, y: int) -> list[int]:
            return page.evaluate(
                "([s,x,y]) => Array.from(document.getElementById('cv'+s).getContext('2d').getImageData(x,y,1,1).data).slice(0,3)",
                [side, x, y],
            )

        def cdata(side: str) -> str:
            return page.evaluate(f"document.getElementById('cv{side}').toDataURL()")

        for side in "AB":
            cb = page.locator("#cv" + side).bounding_box()
            check(
                f"compare canvas {side} non-zero and square",
                cb and cb["width"] > 200 and abs(cb["width"] - cb["height"]) < 1.5,
                str(cb),
            )
        ba, bb2 = (
            page.locator("#cvA").bounding_box(),
            page.locator("#cvB").bounding_box(),
        )
        check(
            "boards sit side by side",
            ba
            and bb2
            and bb2["x"] > ba["x"] + ba["width"] - 2
            and abs(ba["y"] - bb2["y"]) < 2,
            f"{ba} {bb2}",
        )
        check("step 0: identical pixels", cdata("A") == cdata("B"))
        check(
            "step 0: grey block on both",
            cpx("A", 11, 11) == GREY and cpx("B", 11, 11) == GREY,
        )
        for _ in range(3):
            page.click("#c-next")
        check("same moves up to step 2: still identical", cdata("A") == cdata("B"))
        check("divergence marker exists with size", page.locator("#cdiv").is_visible())
        sb, mb = (
            page.locator("#cstrip").bounding_box(),
            page.locator("#cdiv").bounding_box(),
        )
        check(
            "marker inside the strip",
            sb
            and mb
            and mb["x"] >= sb["x"] - 4
            and mb["x"] + mb["width"] <= sb["x"] + sb["width"] + 4
            and mb["width"] >= 4,
            f"{sb} {mb}",
        )
        page.click("#c-first")
        page.click("#c-jump")
        check(
            "jump moves the shared scrubber to step 3",
            page.inner_text("#c-pos") == "step 3",
            page.inner_text("#c-pos"),
        )
        check(
            "both boards show step 3",
            all(
                "step 3" in page.locator(f"#cc-{x} [data-role=stepname]").inner_text()
                for x in "AB"
            ),
        )
        check(
            "step 3: A (21,21) still black",
            cpx("A", 21, 21) == BLACK,
            str(cpx("A", 21, 21)),
        )
        check(
            "step 3: B (21,21) yellow",
            cpx("B", 21, 21) == YELLOW,
            str(cpx("B", 21, 21)),
        )
        check("step 3: pixels differ", cdata("A") != cdata("B"))
        check(
            "moves and sources shown per side",
            "ACTION2" in page.locator("#cc-A [data-role=move]").inner_text()
            and "ACTION3" in page.locator("#cc-B [data-role=move]").inner_text()
            and "model call" in page.locator("#cc-B [data-role=src]").inner_text(),
        )
        check(
            "B reply shown",
            "Try ACTION3" in page.locator("#cc-B [data-role=reply]").inner_text(),
        )
        check(
            "A change line says nothing changed",
            "nothing changed" in page.locator("#cc-A [data-role=change]").inner_text(),
            page.locator("#cc-A [data-role=change]").inner_text(),
        )
        check(
            "B change line counts cells",
            "16 cells" in page.locator("#cc-B [data-role=change]").inner_text(),
        )
        page.click("#cdiv")
        check(
            "clicking the marker also jumps",
            page.inner_text("#c-pos") == "step 3",
            page.inner_text("#c-pos"),
        )
        check(
            "nobody has ended yet",
            not any(
                "ended" in page.locator(f"#cc-{x} [data-role=stepname]").inner_text()
                for x in "AB"
            ),
        )
        page.screenshot(path=str(SHOTS / "compare-diverged.png"), full_page=True)
        page.evaluate(
            "() => { const s = document.getElementById('c-scrub'); s.value = 9; s.dispatchEvent(new Event('input', {bubbles:true})); }"
        )
        held = cdata("B")
        check(
            "B shows ended past its last step",
            page.locator("#cc-B [data-role=stepname]").inner_text() == "ended",
        )
        check(
            "A still playing, not ended",
            page.locator("#cc-A [data-role=stepname]").inner_text() == "step 8",
        )
        page.click("#c-next")
        check("ended board holds its last frame", cdata("B") == held)
        check(
            "ended side shows no move chips",
            page.locator("#cc-B [data-role=move]").count() == 0,
            str(page.locator("#cc-B [data-role=move]").count()),
        )
        page.click("#c-last")
        check("last frame is the longer run's", page.inner_text("#c-pos") == "step 11")
        page.screenshot(path=str(SHOTS / "compare-ended.png"), full_page=True)
        page.click("#c-play")
        page.wait_for_function(
            "document.getElementById('c-pos').textContent === 'start' || document.getElementById('c-pos').textContent !== 'step 11'"
        )
        page.click("#c-play")
        check("compare play toggles", page.inner_text("#c-play") == "Play")
        page.set_input_files("#folder-b", str(FIXTURE))
        page.wait_for_function(
            "document.getElementById('runnames').textContent.includes('B: trace_sample ')||document.getElementById('runnames').textContent.trim().endsWith('B: trace_sample')"
        )
        check(
            "a third load replaces B",
            page.locator("#cmp-table tbody tr").count() == 2
            and "trace_sample_b" not in page.inner_text("#runnames"),
            page.inner_text("#runnames"),
        )
        check(
            "identical runs: no divergence marker",
            page.locator("#cdiv").count() == 0
            and "never play different" in page.inner_text("#c-divtext"),
        )
        page.set_input_files("#folder-b", str(FIXTURE_B))
        page.wait_for_selector("#cdiv")
        # a malformed second run keeps the loaded runs
        page.evaluate(r"() => loadFilesB([new File(['nope\n{}\n'], 'bad.jsonl')])")
        page.wait_for_selector("#error", state="visible")
        check(
            "malformed B shows an error and keeps B",
            "bad.jsonl line 1" in page.inner_text("#error")
            and page.locator("#cmp-table tbody tr").count() == 2,
        )
        check("no page errors in compare", not errors, str(errors))

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
