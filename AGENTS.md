# 🧠 HomeSky — Agents Overview

This document describes the **build, ingest, and visualization agents** used during development and deployment of *HomeSky*, plus how Codex presents their logs and status so you can follow along easily.

## 1️⃣ Agent Roles

### A. Build Agent  
Handles dependency installs, environment setup, and build automation.  
Typical commands:  
- `python -m pip install -r requirements.txt`  
- `python ingest.py --once`  
- `streamlit run visualize_streamlit.py`  
- `pyinstaller --onefile --noconsole -n HomeSky gui.py`

**Codex window behavior:**  
Shows “Installing dependencies…” and package logs; `[Running HomeSky: ingest.py]` headers during script runs; ends with a visible build artifact path (e.g., `dist/HomeSky.exe`).

### B. Ingest Agent  
Connects to the Ambient Weather API, detects cadence, writes new samples to SQLite/Parquet.  
Codex logs lines like:  
`[HomeSky][Ingest] New record inserted: 2025-10-04T14:23Z` and `Cadence ≈ 602s (stable)`.  
Skips duplicates gracefully and retries failed requests.  

**Common-sense cues:**  
Repeated timestamps → station hasn’t posted yet.  
JSON errors → check API keys.

### C. Visualization Agent  
Runs Streamlit dashboard.  
Codex shows “You can now view your Streamlit app…” with an **Open in new tab** button.  
Blank charts = no data yet or wrong range.  
“Disconnected” = refresh tab.

### D. GUI Launcher Agent  
PySimpleGUI helper for quick actions (Fetch, Open Dashboard, Logs).  
Codex prints `[GUI launched]` message when simulated in headless mode.

---

## 2️⃣ Expected Codex Window Updates

| Stage | Example Message | Meaning |
|-------|-----------------|----------|
| 🏗️ Setup | Installing dependencies | pip installs libraries |
| 🌦️ Ingest | `[HomeSky][Ingest] Fetching…` | API request running |
| 💾 Write | `Inserted 1 record` | Data appended successfully |
| 📉 Dashboard | `Streamlit running at :8501` | Visualization active |
| ⚠️ Warning | `Duplicate timestamp skipped` | Normal duplicate handling |
| ❌ Error | `401 Unauthorized` | Check credentials |

Look for `[HomeSky]` prefixes in logs to distinguish app messages from system noise.

---

## 3️⃣ Local Workflow

```
python ingest.py
streamlit run visualize_streamlit.py
pyinstaller --onefile --noconsole -n HomeSky gui.py
```

Logs → `data/logs/ingest.log`  
Database → `data/homesky.sqlite`

---

## 4️⃣ Common-Sense UI Updates

| Element | Behavior | Why |
|----------|-----------|----|
| Spinner | Appears during fetch | Confirms progress |
| Chart fade-in | Smooth transitions | Prevents jarring redraws |
| Auto-refresh | 60-s default | Mirrors station cadence |
| “Stale” badge | API delay | Graceful error |
| Toast “Exported CSV” | Confirms save | User feedback |

---

## 5️⃣ Troubleshooting

| Symptom | Cause | Fix |
|----------|-------|----|
| 401 Unauthorized | Wrong key | Update config.toml |
| No new data | Station offline | Check Ambient site |
| Empty dashboard | Bad date filter | Reset range |
| Duplicate spam | Repeated dateutc | Wait; station delay |
| SQLite locked | Folder permissions | Move data dir |

---

## 6️⃣ Codex for New Users
- **Codex terminal = live console.** Watch the right pane for `[HomeSky]` logs.  
- **Stop** with the red ■ button; logs persist.  
- **Restart** to resume the loop.  
- **Streamlit** may auto-reload; `[rerun]` lines are normal.  

✅ File location: `/HomeSky/Agents.md`  
✅ Maintainer: Alex Balog — THG Geophysics  
✅ Purpose: Quick reference for Codex and local debugging.

---

# 🌤️ README Excerpt Update

## HomeSky

Local weather station dashboard and long-term data collector built for the Ambient Weather WS-2000.

**Docs overview:**
- `Agents.md` → explains how Codex agents and the Streamlit GUI behave.  
- `config.example.toml` → fill in your Ambient API keys and station MAC.  
- `ingest.py` → background collector.  
- `visualize_streamlit.py` → dashboard app.  
- `gui.py` → simple desktop launcher.

**Quick start:**
1. Copy `config.example.toml` → `config.toml`
2. Add your Ambient `apiKey` and `applicationKey`
3. Run:
   ```
   python ingest.py
   streamlit run visualize_streamlit.py
   ```
4. Optional: package into EXE  
   `pyinstaller --onefile --noconsole -n HomeSky gui.py`

**Troubleshooting:**  
See `Agents.md` for detailed logs, agent roles, and Codex terminal expectations.
