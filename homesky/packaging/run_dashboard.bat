@echo off
REM Launch the HomeSky Streamlit dashboard
set SCRIPT_DIR=%~dp0..
cd /d %SCRIPT_DIR%
streamlit run visualize_streamlit.py --server.headless=false
