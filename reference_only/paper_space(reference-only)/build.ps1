param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$templateDir = Join-Path $root "CCN2025_8pager\CCN2025_8pager"
if (Test-Path $templateDir) {
    $env:TEXINPUTS = "$templateDir;$env:TEXINPUTS"
    $env:BSTINPUTS = "$templateDir;$env:BSTINPUTS"
}

$template2026Dir = Join-Path $root "ccn-template-main"
if (Test-Path $template2026Dir) {
    $env:TEXINPUTS = "$template2026Dir;$env:TEXINPUTS"
}

$bibDir = Join-Path $root "bib"
if (Test-Path $bibDir) {
    $env:BIBINPUTS = "$bibDir;$env:BIBINPUTS"
}

if ($Clean) {
    if (Get-Command latexmk -ErrorAction SilentlyContinue) {
        & latexmk -C "main.tex"
        exit $LASTEXITCODE
    }
    Write-Error "latexmk not available. Install Perl or MiKTeX latexmk."
    exit 1
}

$latexmkCommand = Get-Command latexmk -ErrorAction SilentlyContinue
if ($latexmkCommand) {
    & latexmk -pdf -synctex=1 -interaction=nonstopmode "main.tex"
    if ($LASTEXITCODE -eq 0) {
        exit 0
    }
    Write-Warning "latexmk failed; falling back to pdflatex/bibtex."
}

& pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& biber "main"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
exit $LASTEXITCODE
