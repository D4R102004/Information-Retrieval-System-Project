# clean_for_presentation.ps1
# Script para limpiar todos los datos antes de la presentación (Windows)
# Ejecutar: powershell -ExecutionPolicy Bypass -File clean_for_presentation.ps1

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Limpieza del Sistema para Presentación (Windows)          ║" -ForegroundColor Cyan
Write-Host "║  Duración: < 30 segundos                                  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$PROJECT_ROOT = Get-Location

Write-Host "▶ Paso 1: Eliminando índices y vectores..." -ForegroundColor Yellow
Remove-Item -Path "$PROJECT_ROOT/data/index/*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$PROJECT_ROOT/data/processed/*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$PROJECT_ROOT/data/documents.json" -Force -ErrorAction SilentlyContinue
Write-Host "  ✓ Índices eliminados" -ForegroundColor Green

Write-Host ""
Write-Host "▶ Paso 2: Eliminando ChromaDB..." -ForegroundColor Yellow
Remove-Item -Path "$PROJECT_ROOT/data/index/chroma.sqlite3" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$PROJECT_ROOT/data/index/.chroma" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  ✓ ChromaDB eliminado" -ForegroundColor Green

Write-Host ""
Write-Host "▶ Paso 3: Eliminando resultados web previos..." -ForegroundColor Yellow
Remove-Item -Path "$PROJECT_ROOT/data/raw/web/*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  ✓ Resultados web eliminados" -ForegroundColor Green

Write-Host ""
Write-Host "▶ Paso 4: Limpiando logs..." -ForegroundColor Yellow
Remove-Item -Path "$PROJECT_ROOT/logs/*" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  ✓ Logs limpios" -ForegroundColor Green

Write-Host ""
Write-Host "▶ Paso 5: Limpiando caché de Python..." -ForegroundColor Yellow
Get-ChildItem -Path $PROJECT_ROOT -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $PROJECT_ROOT -Filter "*.pyc" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$PROJECT_ROOT/.pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  ✓ Caché de Python limpio" -ForegroundColor Green

Write-Host ""
Write-Host "▶ Paso 6: Verificando estructura de directorios..." -ForegroundColor Yellow
New-Item -Path "$PROJECT_ROOT/data/index" -ItemType Directory -Force | Out-Null
New-Item -Path "$PROJECT_ROOT/data/processed" -ItemType Directory -Force | Out-Null
New-Item -Path "$PROJECT_ROOT/data/raw/web" -ItemType Directory -Force | Out-Null
New-Item -Path "$PROJECT_ROOT/logs" -ItemType Directory -Force | Out-Null
Write-Host "  ✓ Directorios recreados" -ForegroundColor Green

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  ✓ Sistema Limpio - LISTO PARA PRESENTACIÓN               ║" -ForegroundColor Cyan
Write-Host "╠════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║                                                            ║" -ForegroundColor Cyan
Write-Host "║  Próximos pasos:                                           ║" -ForegroundColor Cyan
Write-Host "║  1. python -m ui.app                                       ║" -ForegroundColor Cyan
Write-Host "║  2. Abrir http://localhost:7860                            ║" -ForegroundColor Cyan
Write-Host "║  3. Ir a tab 'Configuration'                               ║" -ForegroundColor Cyan
Write-Host "║  4. Hacer clic 'Load Documents from Crawlers'              ║" -ForegroundColor Cyan
Write-Host "║  5. Seleccionar 'Force re-index corpus'                    ║" -ForegroundColor Cyan
Write-Host "║                                                            ║" -ForegroundColor Cyan
Write-Host "║  Duración indexación: ~3-5 minutos                         ║" -ForegroundColor Cyan
Write-Host "║                                                            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
