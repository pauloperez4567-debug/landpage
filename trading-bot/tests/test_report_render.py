"""Informes para personas: HTML autocontenido con SVG válido, PNG con matplotlib y mini-gráficas de texto."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from desk.backtest import BacktestResult
from desk.clock import SimClock
from desk.optimizer import OptimizationReport, ReportStore, optimize
from desk.report_render import DISCLAIMER, render_html, render_png, sparkline
from desk.tui import _equity_spark, sparkline as tui_sparkline
from synthetic import CoinFlip, HourPattern, pattern_bars

T0 = 1_790_000_000.0
VOID = {"meta", "br", "img", "input", "link", "hr", "col", "area", "base", "wbr", "source"}


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    from desk.config import load_settings

    s = load_settings(env={}, dotenv_path=None, llm_enabled=False,
                      state_dir=str(tmp_path_factory.mktemp("render") / "state"))
    bars = pattern_bars(n=1500)
    rep = optimize("HYPE", "1h", bars, s, strategies=[HourPattern, CoinFlip], clock=SimClock(T0))
    rep.notes.append('nota con <script>alert("x")</script> & "comillas"')
    return rep, rep.result(), bars, s


class _Checker(HTMLParser):
    """Comprueba que las etiquetas abren y cierran en orden (HTML bien formado) y recoge recursos externos."""

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.external = [], [], []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("src", "href") and v and re.match(r"^(https?:)?//", v):
                self.external.append(v)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> con {self.stack[-3:]} abiertas")
            return
        self.stack.pop()


def check_html(text: str) -> _Checker:
    c = _Checker()
    c.feed(text)
    c.close()
    return c


def svgs(text: str) -> list[ET.Element]:
    """Cada ``<svg>...</svg>`` debe ser XML válido (así lo renderiza cualquier navegador o visor)."""
    return [ET.fromstring(m) for m in re.findall(r"<svg\b.*?</svg>", text, flags=re.S)]


def test_html_is_self_contained_valid_and_has_the_charts(case):
    rep, res, bars, _ = case
    text = render_html(rep, res, bars)
    assert text.startswith("<!doctype html>") and '<html lang="es">' in text
    assert '<meta name="viewport"' in text and "<title>Backtest HYPE 1h</title>" in text
    c = check_html(text)
    assert c.errors == [] and c.stack == [] and c.external == []
    assert "http://" not in text and "https://" not in text and "@import" not in text
    charts = [e for e in svgs(text) if e.get("class") == "chart"]
    assert [e.get("aria-label") for e in charts] == ["Precio con entradas y salidas fuera de muestra",
                                                     "Curva de capital fuera de muestra", "Drawdown fuera de muestra"]
    assert all(e.get("viewBox") for e in charts)
    # cada operación fuera de muestra, dibujada (con su descripción al pasar el ratón)
    assert text.count('<g class="trade">') == len(res.trades) > 0
    # tramos walk-forward sombreados y etiquetados
    assert text.count('class="oos"') >= len(rep.get("hour_pattern").folds) and ">T4<" in text


def test_html_honesty_light_dark_and_tables(case):
    rep, res, bars, _ = case
    text = render_html(rep, res, bars)
    assert DISCLAIMER in text and text.count("no garantiza") >= 2
    assert "Dentro de muestra" in text and "Fuera de muestra (lo que cuenta)" in text
    assert "@media (prefers-color-scheme: dark)" in text and ':root[data-theme="dark"]' in text
    assert "background:var(--page)" in text and "max-width:600px" in text
    assert "hour_pattern (ganadora): fuera de muestra" in text and "✔ aprobada" in text and "✘ rechazada" in text
    for e in rep.evaluations:
        for reason in e.reasons:
            assert reason.replace("<", "&lt;").replace(">", "&gt;") in text
    assert "Tramos walk-forward: hour_pattern" in text and "Parámetros elegidos con el entrenamiento: hour=7" in text
    assert "Operaciones fuera de muestra" in text


def test_html_escapes_data(case):
    rep, res, bars, _ = case
    text = render_html(rep, res, bars)
    assert "<script>alert" not in text
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &quot;comillas&quot;" in text
    assert text.count("<script>") == 1                          # sólo el nuestro (lectura al pasar el ratón)


def test_html_with_missing_pieces_never_raises(case):
    rep, res, bars, _ = case
    for args in ((rep, None, bars), (rep, res, None), (None, res, bars), (None, None, None), (rep, None, None)):
        text = render_html(*args)
        c = check_html(text)
        assert c.errors == [] and DISCLAIMER in text
        svgs(text)
    # sin ``result`` usa la curva en memoria de la mejor evaluación
    assert render_html(rep, None, bars).count('<g class="trade">') == len(res.trades)
    # informe leído de disco (sin curvas en memoria) + curva guardada aparte
    back = OptimizationReport.from_dict(rep.to_dict())
    saved = BacktestResult.from_dict(res.to_dict())
    text = render_html(back, saved, bars)
    assert text.count('<g class="trade">') == len(res.trades) and check_html(text).errors == []
    # velas con huecos y basura
    junk = bars.copy()
    junk.iloc[5:20, :] = np.nan
    assert check_html(render_html(rep, res, junk)).errors == []
    assert check_html(render_html(rep, res, pd.DataFrame())).errors == []


def test_long_history_is_summarized(case):
    rep, res, _, _ = case
    big = pattern_bars(n=6000)
    text = render_html(rep, res, big)
    price = next(e for e in svgs(text) if e.get("aria-label", "").startswith("Precio"))
    line = next(p for p in price.iter("{http://www.w3.org/2000/svg}path") if p.get("class") == "price") \
        if price.tag.startswith("{") else next(p for p in price.iter("path") if p.get("class") == "price")
    assert line.get("d").count("L") < 1000                      # resumido a <= 900 puntos
    assert len(text) < 2_000_000


def test_png_starts_with_signature(case):
    rep, res, bars, _ = case
    png = render_png(rep, res, bars)
    assert isinstance(png, bytes) and png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 10_000
    assert render_png(None, None, None).startswith(b"\x89PNG")   # vacío pero válido
    assert render_png(rep, None, bars).startswith(b"\x89PNG")


def test_png_is_none_without_matplotlib_or_on_error(case, monkeypatch):
    rep, res, bars, _ = case
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    assert render_png(rep, res, bars) is None
    monkeypatch.undo()

    class Weird:
        coin = "X"

        def __getattr__(self, name):
            raise RuntimeError("objeto raro")

    assert render_png(Weird(), res, bars) is None


def test_sparkline():
    assert sparkline([1, 2, 3]) == "▁▅█" == tui_sparkline([1, 2, 3])
    assert sparkline([5]) == "" and sparkline([]) == "" and sparkline(None) == ""
    assert sparkline([2, 2, 2]) == "▅▅▅"
    assert sparkline([1, None, "x", float("nan"), True, 3, "2"]) == "▁█▅"
    long = sparkline(range(1000), width=40)
    assert len(long) == 40 and long[0] == "▁" and long[-1] == "█"
    assert list(long) == sorted(long)                           # monótona: resumida, no recortada
    assert len(sparkline(np.sin(np.linspace(0, 6, 500)), width=10)) == 10
    assert set(sparkline(np.random.default_rng(1).normal(size=300))) <= set("▁▂▃▄▅▆▇█")
    # el panel usa ésta (la misma mini-gráfica que en Telegram)
    assert _equity_spark([1.0, 2.0, 3.0]) == "▁▅█"


def test_saved_report_can_be_rendered_again(case, tmp_path):
    rep, res, bars, _ = case
    store = ReportStore(tmp_path)
    store.save(rep, res)
    text = render_html(store.get(rep.id), store.get_result(rep.id), bars)
    assert check_html(text).errors == [] and text.count('<g class="trade">') == len(res.trades)


def test_time_axis_covers_any_span():
    from desk.report_render import _time_ticks

    for span_s in (5 * 3600, 3000 * 3600, 5000 * 14400, 13 * 365 * 86400):
        ticks, step = _time_ticks(1_700_000_000, 1_700_000_000 + span_s)
        assert 2 <= len(ticks) <= 7 and ticks[-1] + step > 1_700_000_000 + span_s   # llega hasta el final
    daily = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": np.linspace(1, 2, 5000), "volume": 1.0},
                         index=pd.date_range("2012-01-01", periods=5000, freq="1D", tz="UTC"))
    text = render_html(None, None, daily)
    assert check_html(text).errors == [] and "2012" in text and "2024" in text
