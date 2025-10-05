# Weather Lake

Weather Lake is a Dark Sky–inspired personal weather lake for Ambient Weather stations. It pairs a lightweight
long-running ingest service with a Streamlit dashboard and a tiny PySimpleGUI launcher so a single power user can
capture, explore, and export rich weather history on Windows. The project is written for Python 3.11+ and is designed
for packaging into a standalone executable via PyInstaller or similar tools.

## Features

- **Long-term capture** – `ingest.py` continuously polls the Ambient Weather Network (AWN) REST API and stores
  observations in SQLite and append-only Parquet files.
- **Fast, beautiful visualization** – `visualize_streamlit.py` provides Dark-Sky-style charts with resampling, derived
  metrics, and export helpers.
- **Desktop first, web ready** – `gui.py` offers Fetch/Open helpers for a Windows desktop workflow, while the Streamlit
  app can be hosted separately with minimal changes.
- **Rich exports** – CSV and Parquet exports are available directly from the dashboard, and data is organized under a
  configurable storage root for other tooling.

## Getting started

1. Install Python 3.11 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `config.example.toml` to `config.toml` and edit it with your Ambient Weather API credentials and preferences.
5. Run the ingest service:
   ```bash
   python ingest.py
   ```
   Use `python ingest.py --once` for a single polling cycle (for Task Scheduler or testing).
6. In another terminal, launch the dashboard:
   ```bash
   streamlit run visualize_streamlit.py
   ```
7. (Optional) Launch the desktop helper:
   ```bash
   python gui.py
   ```

## Packaging on Windows

Use PyInstaller or Briefcase to bundle `gui.py` into a single-file executable. The included `packaging/run_dashboard.bat`
script shows how to start the Streamlit dashboard pointing to the project directory. When packaging, include the `assets`
folder and ensure the `config.toml` and `data` directory are located in a writable location, such as `%APPDATA%/WeatherLake`.

## Data layout

The ingest service respects the `storage` settings in `config.toml`. By default it creates:

- `data/weather.sqlite` – canonical observations table.
- `data/lake.parquet` – append-only Parquet log for analytical tooling.
- `data/logs/ingest.log` – rotating log managed by Loguru.

## Development tips

- Use `explain.py` to print configuration and environment status for troubleshooting.
- Use the `sample_scheduler_task.xml` as a template for Windows Task Scheduler if you prefer to run ingest on an interval.
- Extend `utils/derived.py` to add new calculated metrics; they automatically appear in the dashboard when registered.

## License

This project is released under the MIT License. See `../LICENSE` for details.
