@echo off
if exist "%~dp0ToolLife_ML_Project\ToolLife_ML_Project\venv\Scripts\streamlit.exe" (
    "%~dp0ToolLife_ML_Project\ToolLife_ML_Project\venv\Scripts\streamlit.exe" %*
) else (
    python -m streamlit %*
)
