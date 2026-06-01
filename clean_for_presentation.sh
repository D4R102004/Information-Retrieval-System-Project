#!/bin/bash
# clean_for_presentation.sh
# Script para limpiar todos los datos antes de la presentación
# Ejecutar: bash clean_for_presentation.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Limpieza del Sistema para Presentación                   ║"
echo "║  Duración: < 30 segundos                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(pwd)"

echo "▶ Paso 1: Eliminando índices y vectores..."
rm -rf "$PROJECT_ROOT/data/index"/*
rm -rf "$PROJECT_ROOT/data/processed"/*
rm -f "$PROJECT_ROOT/data/documents.json"
echo "  ✓ Índices eliminados"

echo ""
echo "▶ Paso 2: Eliminando ChromaDB..."
rm -f "$PROJECT_ROOT/data/index/chroma.sqlite3"
rm -rf "$PROJECT_ROOT/data/index/.chroma"
echo "  ✓ ChromaDB eliminado"

echo ""
echo "▶ Paso 3: Eliminando resultados web previos..."
rm -rf "$PROJECT_ROOT/data/raw/web"/*
echo "  ✓ Resultados web eliminados"

echo ""
echo "▶ Paso 4: Limpiando logs..."
rm -rf "$PROJECT_ROOT/logs"/*
touch "$PROJECT_ROOT/logs/.gitkeep"
echo "  ✓ Logs limpios"

echo ""
echo "▶ Paso 5: Limpiando caché de Python..."
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.pytest_cache"
echo "  ✓ Caché de Python limpio"

echo ""
echo "▶ Paso 6: Verificando estructura de directorios..."
mkdir -p "$PROJECT_ROOT/data/index"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/raw/web"
mkdir -p "$PROJECT_ROOT/logs"
echo "  ✓ Directorios recreados"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✓ Sistema Limpio - LISTO PARA PRESENTACIÓN               ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Próximos pasos:                                           ║"
echo "║  1. python -m ui.app                                       ║"
echo "║  2. Abrir http://localhost:7860                            ║"
echo "║  3. Ir a tab 'Configuration'                               ║"
echo "║  4. Hacer clic 'Load Documents from Crawlers'              ║"
echo "║  5. Seleccionar 'Force re-index corpus'                    ║"
echo "║                                                            ║"
echo "║  Duración indexación: ~3-5 minutos                         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
