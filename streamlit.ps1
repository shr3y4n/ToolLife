$venvStreamlit = Join-Path $PSScriptRoot "ToolLife_ML_Project\ToolLife_ML_Project\venv\Scripts\streamlit.exe"
if (Test-Path $venvStreamlit) {
    & $venvStreamlit @args
} else {
    python -m streamlit @args
}
