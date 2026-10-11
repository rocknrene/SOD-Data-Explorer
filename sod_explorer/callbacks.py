"""Dash callbacks.

Callbacks translate interface events into calls to the parsing, source,
analysis, and plotting modules; they contain no scientific logic of their
own. Tables are held in a server-side cache and ``dcc.Store`` components
hold only their keys (see :func:`to_store`); provenance records, which are
small, are held in the stores directly.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html, no_update

from . import provenance
from .analysis import (
    MergeError,
    expedition_values,
    filter_expeditions,
    merge_by_depth,
    merge_depth_column,
    paired_rows,
)
from .columns import depth_columns, find_depth_column, measurement_columns
from .parsing import ParseError, decode_upload, parse_file, resolve_lithology_columns, summarize_site
from .plotting import (
    correlation_figure,
    depth_log_figure,
    dual_axis_figure,
    empty_figure,
    heatmap_figure,
    histogram_figure,
    line_figure,
    scatter_figure,
    smoothed_figure,
    tracks_figure,
)
from .sources import pangaea, routing
from .sources.common import SourceError
from .theme import CARD, DATA_FONT, DROPDOWN, FONT, LABEL

TABLE_PREVIEW_ROWS = 5000
TABLE_PAGE_SIZE = 25

# Largest table accepted from an archive request. Larger requests exhaust the
# memory of small hosting instances and cannot be plotted usefully in a browser.
MAX_FETCH_ROWS = 300_000

# Report types whose whole-expedition requests are too large to retrieve
# (RGB image color data run to millions of rows per expedition).
SITE_REQUIRED_REPORTS = {"rgb"}

# Loaded tables are kept in server memory, and the browser holds only a random
# key for each. Sending whole tables between browser and server as JSON on every
# interaction made chart changes slow for large tables. The cache is per
# process, so the server must run a single process (threads are fine; see the
# Dockerfile). Keys are random UUIDs, so one user's tables cannot be
# reached from another session. The least recently used tables are evicted
# when either limit below is exceeded.
CACHE_MAX_TABLES = 48
CACHE_MAX_BYTES = 200 * 1024 ** 2
_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
_cache_bytes: dict[str, int] = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Table cache and small helpers
# ---------------------------------------------------------------------------

def to_store(df: pd.DataFrame | None) -> str | None:
    """Put a table in the server cache and return the key to hold in a ``dcc.Store``."""
    if df is None:
        return None
    key = uuid.uuid4().hex
    table = df.copy()
    size = int(table.memory_usage(index=True, deep=False).sum())
    with _cache_lock:
        _cache[key] = table
        _cache_bytes[key] = size
        while len(_cache) > 1 and (len(_cache) > CACHE_MAX_TABLES
                                   or sum(_cache_bytes.values()) > CACHE_MAX_BYTES):
            oldest, _ = _cache.popitem(last=False)
            _cache_bytes.pop(oldest, None)
    return key


def from_store(key: str | None) -> pd.DataFrame | None:
    """Return a copy of the cached table for a ``dcc.Store`` value.

    Returns ``None`` if no table was stored, or if the table has left the
    cache (the server restarted, or many newer tables were loaded); the
    interface then behaves as if nothing had been loaded. A copy is returned
    so that a callback cannot alter the cached table.
    """
    if not key:
        return None
    with _cache_lock:
        df = _cache.get(key)
        if df is not None:
            _cache.move_to_end(key)
    return None if df is None else df.copy()


def _positive(value, default: float) -> float:
    """``value`` as a positive number, or ``default`` if it is empty, invalid, or not positive."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _active(mode, merged, a, b):
    """Select the table (or record) shown in the current view: merged, dataset A, or dataset B."""
    return {"a": a, "b": b}.get(mode, merged)


def _options(columns):
    """Dropdown options whose labels and values are the given column names."""
    return [{"label": str(c), "value": c} for c in columns]


def _message(text, kind="info"):
    """Short status line colored by kind: ``ok``, ``warn``, ``error``, or neutral."""
    color = {"ok": "var(--accent2)", "error": "var(--danger)", "warn": "var(--warn)"}.get(kind, "var(--muted)")
    return html.Div(text, style={"color": color, "fontSize": "10px", "fontFamily": FONT,
                                 "whiteSpace": "pre-wrap"})


def data_table(df: pd.DataFrame, highlight_comments: bool = False) -> html.Div:
    """Paged table of the first :data:`TABLE_PREVIEW_ROWS` rows.

    The table scrolls in both directions inside one box whose scrollbars are
    always drawn (see the ``.dash-spreadsheet-container`` rules in the page
    stylesheet), so wide tables can be scrolled sideways without a trackpad.
    """
    preview = df.head(TABLE_PREVIEW_ROWS)
    notes = [_row_count_text(len(df))]
    if len(df.columns) > 14:
        notes.append(f"{len(df.columns)} columns; scroll sideways to see all of them")
    conditional = [{"if": {"row_index": "odd"}, "backgroundColor": "var(--bg)"}]
    if highlight_comments:
        conditional += [{"if": {"filter_query": f"{{{c}}} != ''", "column_id": c}, "color": "var(--warn)"}
                        for c in preview.columns if "comment" in str(c).lower()]
    table = dash_table.DataTable(
        data=preview.astype(object).where(preview.notna(), None).to_dict("records"),
        columns=[{"name": str(c), "id": str(c)} for c in preview.columns],
        page_size=TABLE_PAGE_SIZE, page_action="native", sort_action="native", filter_action="native",
        # One scroll container in both directions, so the horizontal scrollbar
        # stays at the bottom of the visible box rather than below the fold.
        style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "420px",
                     "width": "100%", "minWidth": "100%"},
        style_header={"backgroundColor": "var(--bg)", "color": "var(--accent)", "fontWeight": "700",
                      "fontSize": "10px", "border": "1px solid var(--border)"},
        style_cell={"backgroundColor": "var(--panel)", "color": "var(--text)", "fontSize": "11px",
                    "padding": "7px 11px", "border": "1px solid var(--border)", "fontFamily": DATA_FONT,
                    "maxWidth": "160px", "overflow": "hidden", "textOverflow": "ellipsis"},
        style_data_conditional=conditional,
    )
    caption = "; ".join(notes)
    return html.Div([html.Div(caption, style={"color": "var(--muted)", "fontSize": "10px",
                                              "marginBottom": "4px", "fontFamily": FONT}), table])


def _row_count_text(n_rows: int) -> str:
    """Number of rows in a table, and whether the table display is truncated."""
    if n_rows > TABLE_PREVIEW_ROWS:
        return (f"showing the first {TABLE_PREVIEW_ROWS:,} of {n_rows:,} rows; "
                "the download contains all rows")
    return f"{n_rows:,} row" + ("" if n_rows == 1 else "s")


def _format_depth_range(info: dict) -> str:
    """Depth range and scale of a site summary as text, or ``n/a`` if no depth column was found."""
    if "depth_min" not in info:
        return "n/a"
    return f"{info['depth_min']:.2f} to {info['depth_max']:.2f} ({info.get('depth_scale') or 'scale unknown'})"


def metadata_banner(info: dict, manual: dict) -> html.Div:
    """Banner of site metadata; manual entries take precedence over values detected in the file."""
    def field(label, value):
        """One labeled value of the banner."""
        return html.Div([
            html.Div(label, style={"color": "var(--muted)", "fontSize": "9px", "letterSpacing": "1.5px"}),
            html.Div(value, style={"color": "var(--text)", "fontSize": "13px", "fontWeight": "600"}),
        ], style={"marginRight": "20px", "fontFamily": FONT})

    def pick(key, fallback="n/a"):
        """Manual entry if given, otherwise the detected value, otherwise a fallback."""
        return manual.get(key) or info.get(key) or fallback

    lat, lon = manual.get("lat"), manual.get("lon")
    return html.Div([
        html.Div([
            field("EXPEDITION", pick("expedition")),
            field("SITE / HOLE", pick("site_hole")),
            field("LAT / LON", f"{lat}, {lon}" if lat and lon else "n/a"),
            field("WATER DEPTH", f"{manual['water_depth']} m" if manual.get("water_depth") else "n/a"),
            field("CORE RECOVERY", f"{manual['recovery']}%" if manual.get("recovery") else "n/a"),
            field("DEPTH RANGE", _format_depth_range(info)),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Div([
            html.Span(info.get("filename", ""), style={"background": "var(--border)", "padding": "3px 10px",
                                                       "borderRadius": "12px", "fontSize": "11px"}),
            html.Span(info.get("format", ""), style={"background": "var(--accent)", "color": "var(--bg)",
                                                     "padding": "3px 10px", "borderRadius": "12px",
                                                     "fontSize": "11px", "fontWeight": "700"}),
            html.Span(f"{info.get('rows', 0):,} rows", style={"color": "var(--muted)", "fontSize": "11px"}),
        ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "padding": "10px 20px", "background": "var(--panel)", "flexWrap": "wrap", "gap": "8px",
              "borderBottom": "1px solid var(--border)", "fontFamily": FONT})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_callbacks(app) -> None:
    """Attach all callbacks to a Dash application."""
    _register_global(app)
    _register_shipboard(app)
    for ds in ("a", "b"):
        _register_dataset(app, ds)
    _register_legacy(app)


def _register_global(app):
    """Callbacks shared by both tabs: color theme and tab visibility."""
    app.clientside_callback(
        """
        function(n, stored) {
            const themes = %s;
            let theme = stored === "light" ? "light" : "dark";
            if (n > 0) { theme = theme === "light" ? "dark" : "light"; }
            const vars = themes[theme];
            Object.entries(vars).forEach(([k, v]) => document.documentElement.style.setProperty(k, v));
            document.body.style.backgroundColor = vars["--bg"];
            document.body.style.color = vars["--text"];
            return theme;
        }
        """ % _theme_json(),
        Output("theme-store", "data"),
        Input("theme-toggle", "n_clicks"),
        State("theme-store", "data"),
    )

    @app.callback(Output("theme-toggle", "children"), Input("theme-store", "data"))
    def theme_button_label(theme):
        """Label the theme button with the theme it will switch to."""
        return "Dark mode" if theme == "light" else "Light mode"

    @app.callback(Output("shipboard-container", "style"), Output("legacy-container", "style"),
                  Input("main-tabs", "value"))
    def show_tab(tab):
        """Show the selected tab and hide the other; both stay mounted, so their state persists."""
        shown, hidden = {"display": "block"}, {"display": "none"}
        return (shown, hidden) if tab == "shipboard" else (hidden, shown)


def _theme_json() -> str:
    """CSS variable assignments of every theme, as JSON for the client-side theme callback."""
    import json

    from .theme import THEMES, css_variables
    return json.dumps({name: css_variables(name) for name in THEMES})


# ---------------------------------------------------------------------------
# Shipboard tab
# ---------------------------------------------------------------------------

def _register_shipboard(app):
    """Callbacks of the Shipboard tab."""
    @app.callback(Output("store-df", "data"), Output("store-meta", "data"),
                  Output("store-site-info", "data"), Output("upload-status", "children"),
                  Input("upload", "contents"), State("upload", "filename"), prevent_initial_call=True)
    def load_file(contents, filename):
        """Parse the uploaded data file and store the table, its metadata, and its site summary."""
        try:
            df, meta = parse_file(decode_upload(contents), filename)
        except ParseError as exc:
            return None, None, None, _message(f"Error: {exc}", "error")
        meta["columns"] = [str(c) for c in df.columns]
        return to_store(df), meta, summarize_site(df, meta), _message(f"Loaded {filename}", "ok")

    @app.callback(Output("store-litho", "data"), Output("litho-badge", "children"),
                  Input("upload-litho", "contents"), State("upload-litho", "filename"))
    def load_lithology(contents, filename):
        """Parse the lithology file and store its standardized interval table and depth scale."""
        if not contents:
            return None, _message("No lithology file loaded.")
        try:
            df, _ = parse_file(decode_upload(contents), filename)
            table, info = resolve_lithology_columns(df)
        except ParseError as exc:
            return None, html.Pre(str(exc), style={"color": "var(--danger)", "fontSize": "9px",
                                                   "whiteSpace": "pre-wrap", "fontFamily": FONT})
        text = (f"{filename}: {len(table)} intervals, {table['top_depth'].min():.2f} to "
                f"{table['bottom_depth'].max():.2f} ({info['depth_scale'] or 'scale unknown'})")
        if info["intervals_removed"]:
            text += f"; {info['intervals_removed']} invalid intervals removed"
        return {"table": to_store(table), "depth_scale": info["depth_scale"]}, _message(text, "ok")

    @app.callback(Output("meta-banner", "children"),
                  Input("store-site-info", "data"),
                  Input("meta-expedition", "value"), Input("meta-site-hole", "value"),
                  Input("meta-lat", "value"), Input("meta-lon", "value"),
                  Input("meta-water-depth", "value"), Input("meta-recovery", "value"))
    def update_banner(info, expedition, site_hole, lat, lon, water_depth, recovery):
        """Rebuild the site-metadata banner when a file is loaded or a manual entry changes."""
        if not info:
            return html.Div("Upload a file to see site metadata.",
                            style={"color": "var(--muted)", "fontSize": "11px", "padding": "10px 20px"})
        manual = dict(expedition=expedition, site_hole=site_hole, lat=lat, lon=lon,
                      water_depth=water_depth, recovery=recovery)
        return metadata_banner(info, manual)

    @app.callback(Output("depth-col", "options"), Output("depth-col", "value"),
                  Output("x-col", "options"), Output("x-col", "value"),
                  Output("y-col", "options"), Output("y-col", "value"),
                  Output("color-col", "options"), Output("color-col", "value"),
                  Output("depth-curves", "options"), Output("depth-curves", "value"),
                  Input("store-df", "data"))
    def set_column_options(data):
        """Fill the column menus and choose defaults: detected depth column, first measurement columns."""
        df = from_store(data)
        if df is None:
            return [], None, [], None, [], None, [], None, [], []
        numeric = df.select_dtypes(include="number").columns.tolist()
        depths = depth_columns(df)
        measurements = measurement_columns(df)
        depth = find_depth_column(df)
        depth_opts = _options(depths + [c for c in numeric if c not in depths])
        all_opts = _options(df.columns)
        x_default = measurements[0] if measurements else (numeric[0] if numeric else None)
        return (depth_opts, depth, all_opts, x_default, all_opts, depth or x_default,
                [{"label": "None", "value": "None"}] + all_opts, "None",
                _options(measurements), measurements[:3])

    @app.callback(Output("depth-lbl", "style"), Output("depth-col", "style"),
                  Output("x-lbl", "style"), Output("x-col", "style"),
                  Output("y-lbl", "style"), Output("y-col", "style"),
                  Output("color-lbl", "style"), Output("color-col", "style"),
                  Output("curves-lbl", "style"), Output("depth-curves", "style"),
                  Output("depthlog-options", "style"), Output("axis-options", "style"),
                  Input("chart-type", "value"))
    def toggle_controls(chart_type):
        """Show only the controls that apply to the selected chart type."""
        hide = {"display": "none"}
        label, drop = dict(LABEL), dict(DROPDOWN)
        visible = {
            "depthlog": {"depth", "curves", "depthlog"},
            "scatter": {"x", "y", "color", "axis"},
            "line": {"x", "y", "axis"},
            "histogram": {"x"},
            "heatmap": set(),
        }.get(chart_type, set())
        out = []
        for name in ("depth", "x", "y", "color"):
            out += [label, drop] if name in visible else [hide, hide]
        out += [label, {}] if "curves" in visible else [hide, hide]
        out += [{} if "depthlog" in visible else hide, {} if "axis" in visible else hide]
        return out

    @app.callback(Output("kpi-bar", "children"), Input("store-df", "data"))
    def summary_cards(data):
        """Mean, minimum, maximum, and count of the first six measurement columns."""
        df = from_store(data)
        if df is None:
            return [html.Span("Upload a file to begin.", style={"color": "var(--muted)", "fontSize": "12px"})]
        cards = []
        for column in measurement_columns(df)[:6]:
            values = df[column].dropna()
            if values.empty:
                continue
            cards.append(html.Div([
                html.Div(column, style={"color": "var(--muted)", "fontSize": "9px", "letterSpacing": "1px"}),
                html.Div(f"mean {values.mean():.4g}", style={"color": "var(--text)", "fontSize": "15px",
                                                             "fontWeight": "700"}),
                html.Div(f"min {values.min():.4g}  max {values.max():.4g}  n {len(values):,}",
                         style={"color": "var(--muted)", "fontSize": "9px"}),
            ], style={**CARD, "minWidth": "120px", "padding": "8px 12px", "fontFamily": FONT}))
        return cards

    @app.callback(Output("main-chart", "figure"),
                  Input("store-df", "data"), Input("store-litho", "data"), Input("chart-type", "value"),
                  Input("depth-col", "value"), Input("x-col", "value"), Input("y-col", "value"),
                  Input("color-col", "value"), Input("depth-curves", "value"),
                  Input("overlay-opts", "value"), Input("axis-opts", "value"),
                  Input("gap-threshold", "value"), Input("theme-store", "data"))
    def update_chart(data, litho, chart_type, depth, x, y, color, curves, overlays, axis_opts,
                     gap_threshold, theme):
        """Draw the Shipboard chart for the selected chart type, columns, and overlays."""
        df = from_store(data)
        if df is None:
            return empty_figure(theme=theme)
        overlays, invert = overlays or [], "invert_y" in (axis_opts or [])
        if chart_type == "depthlog":
            if not depth:
                return empty_figure("No depth column was detected; select one", theme)
            lithology = from_store(litho["table"]) if litho else None
            return depth_log_figure(
                df, depth, curves or [], lithology=lithology,
                lithology_scale=litho.get("depth_scale") if litho else None,
                show_gaps="gaps" in overlays, show_flags="flags" in overlays,
                show_core_tops="core_tops" in overlays,
                gap_threshold_m=_positive(gap_threshold, default=5.0), theme=theme)
        if chart_type == "heatmap":
            return heatmap_figure(df, theme)
        if chart_type == "histogram" and x:
            return histogram_figure(df, x, theme)
        if chart_type == "scatter" and x and y:
            return scatter_figure(df, x, y, color, theme, invert)
        if chart_type == "line" and x and y:
            return line_figure(df, x, y, theme, invert)
        return empty_figure("Select axes to plot", theme)

    @app.callback(Output("table-container", "children"), Output("row-count", "children"),
                  Input("store-df", "data"))
    def update_table(data):
        """Show the first rows of the uploaded table, with non-empty comment cells highlighted."""
        df = from_store(data)
        if df is None:
            return _message("No data loaded."), ""
        return data_table(df, highlight_comments=True), ""


# ---------------------------------------------------------------------------
# Legacy Data: datasets A and B
# ---------------------------------------------------------------------------

def _pangaea_choices(ds: str, candidates: list[dict]) -> html.Div:
    """Buttons for the PANGAEA datasets that match a request; selecting one loads it."""
    return html.Div([
        html.Div("Select a dataset to load:", style={"color": "var(--muted)", "fontSize": "9px"}),
        *[html.Button(c["label"], id={"type": "pe-pick", "ds": ds, "pid": c["value"]}, n_clicks=0,
                      style={"display": "block", "width": "100%", "textAlign": "left", "background": "none",
                             "border": "none", "borderBottom": "1px solid var(--border)",
                             "color": "var(--accent3)", "cursor": "pointer", "fontSize": "10px",
                             "padding": "4px 0", "fontFamily": FONT})
          for c in candidates[:8]],
    ])


def _merge_cleared(ds: str) -> tuple:
    """Clear a previous merge when dataset A or B is replaced, so a stale result is never shown."""
    return None, html.Span(f"Dataset {ds.upper()} changed; click Merge datasets to merge again.",
                           style={"color": "var(--warn)"})


def _register_dataset(app, ds: str):
    """Callbacks of one dataset panel (``a`` or ``b``) of the Legacy Data tab."""
    @app.callback(Output(f"pe-{ds}-upload-panel", "style"), Output(f"pe-{ds}-database-panel", "style"),
                  Input(f"pe-{ds}-source", "value"))
    def toggle_source(source):
        """Show the archive controls or the upload control, according to the selected source."""
        hidden = {"display": "none"}
        return ({}, hidden) if source == "upload" else (hidden, {})

    @app.callback(Output(f"pe-store-{ds}", "data"), Output(f"pe-prov-{ds}", "data"),
                  Output(f"pe-{ds}-upload-status", "children"),
                  Output("pe-merged-store", "data", allow_duplicate=True),
                  Output("pe-status-merged", "children", allow_duplicate=True),
                  Input(f"pe-{ds}-upload", "contents"), State(f"pe-{ds}-upload", "filename"),
                  prevent_initial_call=True)
    def load_upload(contents, filename):
        """Parse an uploaded file and store the table with an upload provenance record."""
        if not contents:
            return no_update, no_update, no_update, no_update, no_update
        try:
            df, meta = parse_file(decode_upload(contents), filename)
        except ParseError as exc:
            return None, None, _message(f"Error: {exc}", "error"), no_update, no_update
        text = f"{filename}: {len(df):,} rows"
        if meta.get("zip_member"):
            text += f"; read {meta['zip_member']} from the archive"
            if meta.get("zip_other_members"):
                text += f" ({len(meta['zip_other_members'])} other table file(s) not used)"
        return to_store(df), provenance.upload_record(meta, df), _message(text, "ok"), *_merge_cleared(ds)

    # The Site and Hole menus are filled in the browser from the reference
    # table (held in "pe-ref-store"), so they update immediately even while
    # the server is busy with a long archive request.
    app.clientside_callback(
        """
        function(expedition, ref) {
            const entries = (ref && expedition && ref[expedition]) || [];
            const sites = entries.map(e => ({label: e[0], value: e[0]}));
            return [sites, null, [], null];
        }
        """,
        Output(f"pe-{ds}-site", "options"), Output(f"pe-{ds}-site", "value"),
        Output(f"pe-{ds}-hole", "options", allow_duplicate=True),
        Output(f"pe-{ds}-hole", "value", allow_duplicate=True),
        Input(f"pe-{ds}-exp", "value"), State("pe-ref-store", "data"), prevent_initial_call=True)

    app.clientside_callback(
        """
        function(site, expedition, ref) {
            const entries = (ref && expedition && ref[expedition]) || [];
            const entry = entries.find(e => e[0] === site);
            const holes = entry ? entry[1].map(h => ({label: h[1], value: h[0]})) : [];
            return [holes, null];
        }
        """,
        Output(f"pe-{ds}-hole", "options", allow_duplicate=True),
        Output(f"pe-{ds}-hole", "value", allow_duplicate=True),
        Input(f"pe-{ds}-site", "value"), State(f"pe-{ds}-exp", "value"), State("pe-ref-store", "data"),
        prevent_initial_call=True)

    @app.callback(Output(f"pe-store-{ds}", "data", allow_duplicate=True),
                  Output(f"pe-prov-{ds}", "data", allow_duplicate=True),
                  Output(f"pe-{ds}-db-status", "children"),
                  Output(f"pe-{ds}-db-results", "children"),
                  Output("pe-merged-store", "data", allow_duplicate=True),
                  Output("pe-status-merged", "children", allow_duplicate=True),
                  Input(f"pe-{ds}-fetch", "n_clicks"),
                  State(f"pe-{ds}-report", "value"), State(f"pe-{ds}-exp", "value"),
                  State(f"pe-{ds}-site", "value"), State(f"pe-{ds}-hole", "value"),
                  prevent_initial_call=True)
    def fetch_from_archive(n_clicks, report, expedition, site, hole):
        """Retrieve the requested report from the archive chosen by the routing rules."""
        unchanged = (no_update, no_update)
        if not expedition:
            return no_update, no_update, _message("Select a Leg or Expedition.", "warn"), "", *unchanged
        if not report:
            return no_update, no_update, _message("Select a report type.", "warn"), "", *unchanged
        if report in SITE_REQUIRED_REPORTS and not site:
            return (no_update, no_update,
                    _message("This data type is too large to retrieve for a whole Leg or Expedition. "
                             "Select a Site (and preferably a Hole).", "warn"), "", *unchanged)
        outcome = routing.fetch(report, str(expedition), site or "", hole or "")
        if outcome.df is not None:
            # The archives report no row count before download, so the limit can only be
            # applied once the table has been retrieved.
            if len(outcome.df) > MAX_FETCH_ROWS:
                return (no_update, no_update,
                        _message(f"The request returned {len(outcome.df):,} rows, more than the "
                                 f"{MAX_FETCH_ROWS:,} this application can display. Select a Site "
                                 "or Hole to narrow it.", "error"), "", *unchanged)
            record = provenance.new_record(outcome.source, outcome.df, label=outcome.message)
            return (to_store(outcome.df), record, _message(outcome.message, "ok"), "",
                    *_merge_cleared(ds))
        if outcome.candidates:
            return (no_update, no_update, _message(outcome.message),
                    _pangaea_choices(ds, outcome.candidates), *unchanged)
        return no_update, no_update, _message(outcome.message, "error"), "", *unchanged

    @app.callback(Output(f"pe-store-{ds}", "data", allow_duplicate=True),
                  Output(f"pe-prov-{ds}", "data", allow_duplicate=True),
                  Output(f"pe-{ds}-db-status", "children", allow_duplicate=True),
                  Output("pe-merged-store", "data", allow_duplicate=True),
                  Output("pe-status-merged", "children", allow_duplicate=True),
                  Input({"type": "pe-pick", "ds": ds, "pid": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def pick_pangaea(clicks):
        """Download the PANGAEA dataset the user selected from the candidates."""
        trigger = ctx.triggered_id
        if not trigger or not any(clicks or []):
            return no_update, no_update, no_update, no_update, no_update
        try:
            df, source = pangaea.fetch_dataset(trigger["pid"])
        except SourceError as exc:
            return no_update, no_update, _message(str(exc), "error"), no_update, no_update
        message = f"PANGAEA dataset {trigger['pid']} ({len(df):,} rows)"
        return (to_store(df), provenance.new_record(source, df, label=message), _message(message, "ok"),
                *_merge_cleared(ds))


# ---------------------------------------------------------------------------
# Legacy Data: merge, filters, chart, table, export
# ---------------------------------------------------------------------------

def _merge_status(report: dict) -> list:
    """Merge summary for display: pairs formed, tolerance, grouping, duplicates removed, warnings."""
    t_ok = {"color": "var(--accent2)", "fontWeight": "600"}
    lines = [
        html.Span("Merged  ", style=t_ok),
        html.Span(f"{report['matched']:,} of {report['rows_a_with_depth']:,} A samples paired "
                  f"(tolerance {report['tolerance_m'] * 100:g} cm; grouping by {report['grouping']}"
                  + (f"; median offset {report['median_abs_offset_m'] * 100:.2g} cm"
                     if report.get("median_abs_offset_m") is not None else "") + ")"),
    ]
    if report.get("duplicate_matches_removed"):
        lines.append(html.Div(f"{report['duplicate_matches_removed']:,} duplicate pairings removed "
                              "(each B sample used once).", style={"fontSize": "10px"}))
    for warning in report.get("warnings", []):
        lines.append(html.Div(f"Note: {warning}", style={"color": "var(--warn)", "fontSize": "10px"}))
    return lines


def _register_legacy(app):
    """Callbacks of the Legacy Data tab that act on the loaded datasets."""
    @app.callback(Output("pe-depth-a", "options"), Output("pe-depth-a", "value"),
                  Output("pe-depth-b", "options"), Output("pe-depth-b", "value"),
                  Input("pe-store-a", "data"), Input("pe-store-b", "data"))
    def depth_options(data_a, data_b):
        """Offer each dataset's numeric columns as merge depth, depth columns first."""
        out = []
        for data in (data_a, data_b):
            df = from_store(data)
            if df is None:
                out += [[], None]
                continue
            numeric = df.select_dtypes(include="number").columns.tolist()
            depths = depth_columns(df)
            out += [_options(depths + [c for c in numeric if c not in depths]), find_depth_column(df)]
        return out

    @app.callback(Output("pe-status-a", "children"), Output("pe-status-b", "children"),
                  Input("pe-store-a", "data"), Input("pe-store-b", "data"),
                  Input("pe-prov-a", "data"), Input("pe-prov-b", "data"))
    def status_cards(data_a, data_b, prov_a, prov_b):
        """Row and column counts and source of each loaded dataset."""
        def card(data, record, label, color):
            """Status text of one dataset."""
            df = from_store(data)
            if df is None:
                return [html.Span(f"{label}: ", style={"color": color}), "no data loaded"]
            source = (record or {}).get("source", {}).get("name", "")
            return [html.Span(f"{label}  ", style={"color": color, "fontWeight": "600"}),
                    html.Span(f"{len(df):,} rows x {len(df.columns)} columns"),
                    html.Div(source, style={"fontSize": "10px"})]
        return (card(data_a, prov_a, "Dataset", "var(--accent)"),
                card(data_b, prov_b, "Dataset B", "var(--accent3)"))

    @app.callback(Output("pe-merged-store", "data"), Output("pe-prov-merged", "data"),
                  Output("pe-status-merged", "children"), Output("pe-merge-busy", "children"),
                  Input("pe-merge-btn", "n_clicks"),
                  State("pe-store-a", "data"), State("pe-store-b", "data"),
                  State("pe-prov-a", "data"), State("pe-prov-b", "data"),
                  State("pe-depth-a", "value"), State("pe-depth-b", "value"),
                  State("pe-tolerance", "value"), State("pe-merge-opts", "value"),
                  prevent_initial_call=True)
    def merge(n_clicks, data_a, data_b, prov_a, prov_b, depth_a, depth_b, tolerance_cm, options):
        """Merge datasets A and B by depth and store the result with its merge report and provenance."""
        dfa, dfb = from_store(data_a), from_store(data_b)
        if dfa is None or dfb is None:
            return no_update, no_update, "Load both datasets before merging.", ""
        depth_a = depth_a if depth_a in dfa.columns else find_depth_column(dfa)
        depth_b = depth_b if depth_b in dfb.columns else find_depth_column(dfb)
        if not depth_a or not depth_b:
            return no_update, no_update, "Select a depth column for each dataset.", ""
        options = options or []
        try:
            merged, report = merge_by_depth(
                dfa, dfb, depth_a, depth_b,
                tolerance_m=float(tolerance_cm if tolerance_cm is not None else 2) / 100.0,
                allow_mixed_scales="mixed_scales" in options, one_to_one="one_to_one" in options,
                across_holes="across_holes" in options,
                assumed_scale_a=_source_scale(prov_a), assumed_scale_b=_source_scale(prov_b))
        except MergeError as exc:
            return None, None, html.Span(f"Merge not performed: {exc}", style={"color": "var(--danger)"}), ""
        report_dict = report.to_dict()
        return (to_store(merged), provenance.merge_record(prov_a, prov_b, report_dict, merged),
                _merge_status(report_dict), "")

    @app.callback(Output("pe-exp-filter", "options"), Output("pe-exp-filter", "value"),
                  Output("pe-exp-hint", "children"),
                  Input("pe-view-mode", "value"), Input("pe-merged-store", "data"),
                  Input("pe-store-a", "data"), Input("pe-store-b", "data"))
    def expedition_options(mode, merged, data_a, data_b):
        """List the Expeditions present in the displayed table; all are selected initially."""
        df = from_store(_active(mode, merged, data_a, data_b))
        values = expedition_values(df) if df is not None else []
        hint = "" if values or df is None else "No Expedition column in this dataset; all rows are shown."
        return [{"label": f" {v}", "value": v} for v in values], values, hint

    @app.callback(Output("pe-exp-filter", "value", allow_duplicate=True),
                  Input("pe-exp-all-none", "n_clicks"),
                  State("pe-exp-filter", "options"), State("pe-exp-filter", "value"),
                  prevent_initial_call=True)
    def toggle_all_expeditions(n_clicks, options, current):
        """Select all Expeditions, or none if all are already selected."""
        values = [o["value"] for o in options or []]
        return [] if set(current or []) == set(values) else values

    @app.callback(Output("pe-merge-section", "style"),
                  Output("pe-status-b", "style"), Output("pe-status-merged", "style"),
                  Output("pe-depth-a-container", "style"), Output("pe-depth-b-container", "style"),
                  Output("pe-ycols-b-container", "style"), Output("pe-rows-container", "style"),
                  Output("pe-a-label", "children"),
                  Input("pe-view-mode", "value"))
    def view_mode_controls(mode):
        """Show dataset B, the merge settings, and the merge-depth pickers only in merge view.

        These controls have no effect on a single dataset, so they are hidden
        rather than left inactive.
        """
        hidden = {"display": "none"}
        status = {"flex": "1", "background": "var(--panel)", "border": "1px solid var(--border)",
                  "borderRadius": "6px", "padding": "10px 14px", "fontSize": "12px",
                  "color": "var(--muted)", "minWidth": "200px"}
        picker = {"flex": "1 1 180px", "minWidth": "160px"}
        if mode == "merged":
            return ({}, {**status, "marginRight": "8px"}, status, picker, picker,
                    {"flex": "2 1 240px", "minWidth": "160px"}, {"marginTop": "8px"}, "DATASET A")
        return hidden, hidden, hidden, hidden, hidden, hidden, hidden, "DATASET"

    @app.callback(Output("pe-xaxis-lbl", "children"), Output("pe-yaxis-lbl", "children"),
                  Output("pe-ycols-b-lbl", "children"),
                  Input("pe-view-mode", "value"), Input("pe-chart-mode", "value"))
    def axis_labels(mode, chart_mode):
        """Name the column pickers for what the selected chart does with them."""
        merged = mode == "merged"
        if chart_mode == "scatter":
            if merged:
                return "DEPTH (COLORS THE POINTS)", "X AXIS: DATASET A COLUMN", "Y AXIS: DATASET B COLUMN"
            return "DEPTH (COLORS THE POINTS)", "X AXIS, THEN Y AXIS (TWO COLUMNS)", ""
        if chart_mode == "dual":
            if merged:
                return "DEPTH AXIS", "LEFT AXIS: DATASET A COLUMN", "RIGHT AXIS: DATASET B COLUMN"
            return "DEPTH AXIS", "LEFT AXIS, THEN RIGHT AXIS (TWO COLUMNS)", ""
        return ("DEPTH AXIS", "DATASET A COLUMNS" if merged else "COLUMNS",
                "DATASET B COLUMNS")

    @app.callback(Output("pe-xaxis", "options"), Output("pe-xaxis", "value"),
                  Output("pe-yaxis", "options"), Output("pe-yaxis", "value"),
                  Output("pe-ycols-b", "options"), Output("pe-ycols-b", "value"),
                  Input("pe-view-mode", "value"), Input("pe-merged-store", "data"),
                  Input("pe-store-a", "data"), Input("pe-store-b", "data"),
                  State("pe-chart-mode", "value"))
    def axis_options(mode, merged, data_a, data_b, chart_mode):
        """Offer depth columns for the depth axis and measurement columns for the tracks.

        Only numeric measurement columns are offered: plotting a text field
        would assign arbitrary positions to its categories, and plotting an
        identifier (Core, Section, offset) would suggest a physical trend
        where there is none. In merged view the merge depth is the default
        depth axis, and the columns of each dataset are taken from that
        dataset itself, so that a column of B is offered even when few of its
        samples were paired.
        """
        df = from_store(_active(mode, merged, data_a, data_b))
        if df is None:
            return [], None, [], None, [], None
        two = chart_mode in ("scatter", "dual")
        if mode == "merged":
            depths = depth_columns(df)
            depth = merge_depth_column(df) or find_depth_column(df)
            if depth is not None and depth not in depths:
                depths = [depth] + depths
            cols_a = _merged_columns(df, from_store(data_a), "_A")
            cols_b = _merged_columns(df, from_store(data_b), "_B")
            return (_options(depths), depth, _options(cols_a), cols_a[:1 if two else 3],
                    _options(cols_b), cols_b[:1 if two else 3])
        depths = depth_columns(df)
        measures = measurement_columns(df)
        return _options(depths), find_depth_column(df), _options(measures), measures[:2 if two else 3], [], []

    @app.callback(Output("pe-yaxis", "value", allow_duplicate=True),
                  Output("pe-ycols-b", "value", allow_duplicate=True),
                  Input("pe-chart-mode", "value"),
                  State("pe-view-mode", "value"), State("pe-yaxis", "value"), State("pe-ycols-b", "value"),
                  prevent_initial_call=True)
    def trim_columns_for_pair_charts(chart_mode, mode, cols_a, cols_b):
        """Keep only the columns a two-column chart uses, so the pickers show what is plotted."""
        if chart_mode not in ("scatter", "dual"):
            return no_update, no_update
        cols_a, cols_b = list(cols_a or []), list(cols_b or [])
        if mode == "merged":
            return cols_a[:1], cols_b[:1]
        return cols_a[:2], no_update

    @app.callback(Output("pe-rolling-ctrl", "style"), Output("pe-detrend-ctrl", "style"),
                  Input("pe-chart-mode", "value"))
    def toggle_mode_controls(mode):
        """Show the smoothing window or detrending option only for the chart mode that uses it."""
        hidden = {"display": "none"}
        return ({} if mode == "rolling" else hidden), ({} if mode == "scatter" else hidden)

    @app.callback(Output("pe-chart", "figure"),
                  Input("pe-view-mode", "value"), Input("pe-merged-store", "data"),
                  Input("pe-store-a", "data"), Input("pe-store-b", "data"),
                  Input("pe-exp-filter", "value"), Input("pe-exp-filter", "options"),
                  Input("pe-xaxis", "value"), Input("pe-yaxis", "value"), Input("pe-ycols-b", "value"),
                  Input("pe-chart-mode", "value"), Input("pe-rolling-window", "value"),
                  Input("pe-detrend", "value"), Input("pe-rows", "value"), Input("theme-store", "data"))
    def update_chart(mode, merged, data_a, data_b, selected, exp_options, depth, cols_a, cols_b,
                     chart_mode, window, detrend, rows, theme):
        """Draw the Legacy Data chart for the displayed table, selected columns, and chart mode."""
        df = _displayed(mode, merged, data_a, data_b, selected, exp_options, rows)
        if df is None:
            hint = "Click Merge datasets to view the merged data" if mode == "merged" else "Load a dataset"
            return empty_figure(hint, theme)
        if mode == "merged" and rows == "paired" and df.empty:
            return empty_figure("No paired samples to show; increase the tolerance or show all A samples",
                                theme)
        if not depth or depth not in df.columns:
            return empty_figure("Select a depth column", theme)
        df = df.dropna(subset=[depth])
        cols_a = [c for c in (cols_a or []) if c in df.columns]
        cols_b = [c for c in (cols_b or []) if c in df.columns] if mode == "merged" else []
        if not cols_a and not cols_b:
            return empty_figure("Select one or more columns", theme)
        if chart_mode in ("scatter", "dual"):
            if mode != "merged" and len(cols_a) >= 2:
                first, second = cols_a[0], cols_a[1]
            elif cols_a and cols_b:
                first, second = cols_a[0], cols_b[0]
            else:
                return empty_figure("Select one column from each dataset (or two columns)", theme)
            if chart_mode == "scatter":
                return correlation_figure(df, depth, first, second, theme,
                                          detrend="detrend" in (detrend or []))
            return dual_axis_figure(df, depth, first, second, theme)
        if chart_mode == "rolling":
            window_m = _positive(window, default=0.0)
            if window_m <= 0:
                return empty_figure("Enter a positive window length", theme)
            return smoothed_figure(df, depth, cols_a, cols_b, window_m, theme=theme)
        return tracks_figure(df, depth, cols_a, cols_b, theme)

    @app.callback(Output("pe-table-container", "children"),
                  Input("pe-view-mode", "value"), Input("pe-merged-store", "data"),
                  Input("pe-store-a", "data"), Input("pe-store-b", "data"),
                  Input("pe-exp-filter", "value"), Input("pe-exp-filter", "options"),
                  Input("pe-rows", "value"))
    def update_table(mode, merged, data_a, data_b, selected, exp_options, rows):
        """Show the displayed table after the Expedition filter and the paired-rows choice."""
        df = _displayed(mode, merged, data_a, data_b, selected, exp_options, rows)
        if df is None:
            return ""
        return data_table(df)

    @app.callback(Output("pe-download", "data"),
                  Input("pe-download-btn", "n_clicks"),
                  State("pe-view-mode", "value"), State("pe-merged-store", "data"),
                  State("pe-store-a", "data"), State("pe-store-b", "data"),
                  State("pe-prov-merged", "data"), State("pe-prov-a", "data"), State("pe-prov-b", "data"),
                  State("pe-exp-filter", "value"), State("pe-exp-filter", "options"),
                  State("pe-rows", "value"),
                  prevent_initial_call=True)
    def download(n_clicks, mode, merged, data_a, data_b, prov_merged, prov_a, prov_b, selected,
                 exp_options, rows):
        """Export the displayed table, after the Expedition filter, with its provenance and citations."""
        df = from_store(_active(mode, merged, data_a, data_b))
        if df is None:
            return no_update
        record = _active(mode, prov_merged, prov_a, prov_b)
        if exp_options:
            df = filter_expeditions(df, selected)
            record = provenance.add_step(record, "filter_expeditions", selected=selected)
        if mode == "merged" and rows == "paired":
            df = paired_rows(df)
            record = provenance.add_step(record, "keep_paired_rows_only",
                                         note="rows of A without a paired B sample were omitted")
        name = {"a": "dataset_a", "b": "dataset_b"}.get(mode, "merged_a_b")
        return dcc.send_bytes(provenance.export_archive(df, record, basename=f"sod_{name}"),
                              f"sod_{name}.zip")


def _displayed(mode: str, merged: str | None, data_a: str | None, data_b: str | None,
               selected: list[str] | None, exp_options: list[dict] | None,
               rows: str | None) -> pd.DataFrame | None:
    """The table shown in the chart and data table.

    This is the table of the current view, filtered by the Expedition
    checkboxes and, in merged view with "Paired samples only", restricted to
    the rows in which a B sample was paired.
    """
    df = from_store(_active(mode, merged, data_a, data_b))
    if df is None:
        return None
    df = filter_expeditions(df, selected if exp_options else None)
    if mode == "merged" and rows == "paired":
        df = paired_rows(df)
    return df


def _merged_columns(merged: pd.DataFrame, source: pd.DataFrame | None, suffix: str) -> list[str]:
    """Measurement columns of one dataset as they are named in the merged table."""
    if source is not None:
        names = [f"{c}{suffix}" for c in measurement_columns(source)]
        found = [c for c in names if c in merged.columns]
        if found:
            return found
    return [c for c in measurement_columns(merged) if str(c).endswith(suffix)]


def _source_scale(record: dict | None) -> str | None:
    """Depth scale defined by a dataset's source when its headers name none.

    DSDP depths are meters below seafloor ("mbsf"); DSDP predates composite
    depth scales. Other sources state the scale in their headers.
    """
    source_type = ((record or {}).get("source") or {}).get("type", "")
    return "mbsf" if str(source_type).startswith("dsdp") else None
