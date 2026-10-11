# Changelog

## 2.0.0 (unreleased)

Restructured from a single-file application into a tested package.

### GSA testing fixes (testing matrix of 2026-10-08/09)
- Server runs one process with eight threads, so a long archive request no
  longer freezes the page for every user; loaded tables are kept on the
  server and only a key is sent to the browser, so chart changes are fast.
- Site and Hole menus are filled in the browser and update immediately.
- "Working on it" message over the chart and table during fetches, merges,
  and redraws; dot indicator under Merge datasets.
- Merged view: "Paired samples only" (default) or "All samples of A" for the
  chart, table, and download; merge depth is the default depth axis; Dataset
  B columns are always offered; a warning when nothing pairs; DSDP depths are
  labeled mbsf; replacing A or B clears the previous merge.
- Expedition filter removes rows holding any unselected Expedition, so it
  acts on both datasets of a merge.
- Data table pages through up to 5,000 rows (previously the first 200).
- Thicker lines, larger points, chart names ("Depth plots", "Cross-plot (two
  columns)", "Depth plot, dual axis"), axis-picker labels for two-column
  charts, solid lines on the dual-axis plot, clearer chart help text.
- Number fields' +/- buttons no longer clear the value (Dash 4.4 min/max
  issue); default window 10 m, with advice when the window is too short.
- Section offsets ("Top (cm)", "Bottom (cm)") are no longer default columns.
- RGB requires a Site; requests over 300,000 rows are refused with advice.
- Shipboard tab carries an under-development notice; Dataset help lists DSDP,
  ODP, IODP in order.

### Report types (data-type table by L. Childress)
- Report menu rebuilt from the cross-program "SOD Data Types" table: sixteen
  generic names in three categories replace the previous LIMS and DSDP lists.
- New types: downhole temperature, carbonates, gas elements, interstitial
  water, source rock analysis, penetrometer strength, split-core P-wave
  velocity (PWC and PWB combined), color reflectance, RGB, torvane strength.
- DSDP thermal conductivity, Rock-Eval, and penetrometer data are searched on
  PANGAEA, as the table specifies; types DSDP did not measure say so.
- LORE report names are resolved by probing candidates;
  `python -m sod_explorer.sources.check_lore` reports which resolve.
- Removed from the menu: WRMSL, Shore XRF, and the DSDP-native categories.

### PANGAEA-style tables
- Sample labels ("177-1090E-8H-3,130") are expanded into Expedition, Site,
  Hole, and Core columns, validated against the reference table.
- PANGAEA "Depth sed" is recognized as depth below seafloor and preferred
  over composite depth as the default depth axis.

### Reference table
- Site and hole list attributed to L. Childress (personal communication).
- Program and platform assignments checked against the SOD Legacy Data
  Access Quick Start Guide (v1.0); all existing rows agree.
- Added Expeditions 380 and 405 (Chikyu) and 389 (MSP), without sites and
  holes, so they can be selected and routed correctly.

### Interface (review by L. Childress)
- Renamed to SOD (Scientific Ocean Drilling) Explorer; the Python package
  is now `sod_explorer`.
- "Post-Expedition" tab renamed "Legacy Data" and placed first; Shipboard
  moved to the right.
- View choice reduced to "Single dataset" or "Merge two datasets"; dataset
  B, the merge settings, and the merge-depth pickers are shown only in
  merge view.
- Explanatory sidebar text moved behind ⓘ information buttons.
- Hole menu shows `*` alone for unlettered DSDP holes.
- Data tables scroll horizontally within a single visible scroll box.
- DSDP "sample depth (m)" is preferred over "top interval depth (cm)" and
  "top of core depth (m)" when choosing the depth column.

### Scientific corrections
- Merge match count now reports pairs formed; it previously reported the
  number of rows in dataset A.
- Merge pairs samples only within the same hole (or the same site on a
  composite depth scale), refuses mismatched depth scales unless explicitly
  allowed, converts depth units from column headers, and by default uses
  each B sample at most once.
- Correlation p-values and confidence intervals use an effective sample
  size adjusted for lag-1 autocorrelation (Bretherton et al., 1999). The
  previous p-value assumed independent samples. Optional removal of the
  linear depth trend before correlating.
- Correlation matrix and summary cards exclude identifier columns (Core,
  Section, offsets, sample IDs) and depth columns.
- "Recovery gaps" renamed to sampling gaps and evaluated per hole.
- Smoothing uses a depth window (m) evaluated per hole instead of a row
  window across the whole table.
- Depth axes are labeled with the depth column header instead of "mbsf";
  depth-column detection prefers CSF-A and no longer falls back to the first
  column.
- Lines are drawn per hole and never connect samples from different holes.

### Provenance and reproducibility
- Provenance record for every dataset; exports are ZIP archives with
  data, `provenance.json`, and `CITATION.txt` (PANGAEA citations and DOIs).
- Reference table shipped as a documented CSV instead of an embedded blob.
- Pinned `requirements.txt`, `pyproject.toml`, `CITATION.cff`, MIT license,
  test suite (190 tests), and continuous integration.

### Other fixes
- Expedition filter works on LORE and merged tables.
- Header-row metadata recognizes numeric DSDP/ODP site numbers.
- Text decoding no longer masks UTF-16 and Windows-1252 files behind Latin-1.
- LAS read failures report an error instead of raising `NameError`.
- LORE cache stores only successful requests without clearing other entries.
- Excel files are read once rather than up to twenty times.
- Theme toggle respects the stored preference on reload.
- IODP expeditions absent from the reference table are tried against LORE
  and then PANGAEA instead of being assumed to be JOIDES Resolution.
- Removed unused NGDC placeholder, legacy PANGAEA search, and the DSDP NCEI
  downloader that sent a browser User-Agent.
