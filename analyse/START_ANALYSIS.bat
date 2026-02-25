@echo off
setlocal enabledelayedexpansion

:: ================================================================
::         AIv3 PATTERN ANALYSIS - HAUPTANWENDUNG
:: ================================================================
::  Vollständige Analyse-Suite mit GCS-Datenintegration
::  Verwendet reale Marktdaten aus Google Cloud Storage
:: ================================================================

title AIv3 Pattern Analysis System

:: Farben setzen
color 0A

:: Banner anzeigen
:SHOW_BANNER
cls
echo.
echo    ====================================================================
echo                   AIv3 CONSOLIDATION PATTERN ANALYZER
echo    ====================================================================
echo.
echo                     Erweiterte Marktanalyse mit KI
echo                  Vollständige historische Daten aus GCS
echo.
echo    ====================================================================
echo.
timeout /t 2 >nul

:: Variablen initialisieren
set PYTHON_OK=NO
set GCS_OK=NO
set READY=NO

:: Schnelle Systemprüfung
:QUICK_CHECK
python --version >nul 2>&1
if %errorlevel%==0 set PYTHON_OK=YES

if exist "gcs-key.json" (
    set GCS_OK=YES
) else if exist "C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json" (
    echo Kopiere GCS-Credentials...
    copy "C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json" "gcs-key.json" >nul
    set GCS_OK=YES
)

if "%PYTHON_OK%"=="YES" if "%GCS_OK%"=="YES" set READY=YES

:: Hauptmenü
:MAIN_MENU
cls
echo ================================================================
echo                    AIv3 ANALYSIS SYSTEM
echo ================================================================
echo.
if "%READY%"=="YES" (
    echo Status: [BEREIT] System einsatzbereit
) else (
    echo Status: [SETUP] Konfiguration erforderlich
)
echo.
echo ----------------------------------------------------------------
echo                      HAUPTMENÜ
echo ----------------------------------------------------------------
echo.
echo   [1] Schnellanalyse      - 25 Ticker  (~5 Min)
echo   [2] Standardanalyse     - 50 Ticker  (~10 Min)
echo   [3] Erweiterte Analyse  - 100 Ticker (~20 Min)
echo   [4] Grosse Analyse      - 200 Ticker (~45 Min)
echo   [5] Maximum Analyse     - 500 Ticker (~2 Std)
echo.
echo   [C] Benutzerdefiniert   - Eigene Parameter
echo   [A] ALLE Daten          - Kompletter Datensatz
echo.
echo   [S] System-Setup        - Installation/Konfiguration
echo   [H] Hilfe               - Dokumentation
echo   [Q] Beenden
echo.
echo ================================================================
echo.

set /p CHOICE="Ihre Auswahl: "

:: Auswahl verarbeiten
if /i "%CHOICE%"=="Q" goto :EXIT
if /i "%CHOICE%"=="S" goto :SETUP
if /i "%CHOICE%"=="H" goto :HELP
if /i "%CHOICE%"=="C" goto :CUSTOM
if /i "%CHOICE%"=="A" goto :ALL_DATA

:: Numerische Auswahl
if "%CHOICE%"=="1" set TICKERS=25&& goto :START_ANALYSIS
if "%CHOICE%"=="2" set TICKERS=50&& goto :START_ANALYSIS
if "%CHOICE%"=="3" set TICKERS=100&& goto :START_ANALYSIS
if "%CHOICE%"=="4" set TICKERS=200&& goto :START_ANALYSIS
if "%CHOICE%"=="5" set TICKERS=500&& goto :START_ANALYSIS

:: Ungültige Eingabe
echo.
echo [FEHLER] Ungültige Auswahl!
timeout /t 2 >nul
goto :MAIN_MENU

:: ==================== SETUP ====================
:SETUP
cls
echo ================================================================
echo                    SYSTEM SETUP
echo ================================================================
echo.

:: Python prüfen
echo [1/3] Python-Installation prüfen...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [FEHLER] Python nicht gefunden!
    echo   Bitte installieren Sie Python 3.8+ von python.org
    echo.
    pause
    goto :MAIN_MENU
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
    echo   [OK] Python !PY_VER! gefunden
)

:: Pakete installieren
echo.
echo [2/3] Python-Pakete installieren...
if not exist "requirements.txt" (
    (
        echo pandas^>=2.0.0
        echo numpy^>=1.24.0
        echo matplotlib^>=3.7.0
        echo seaborn^>=0.12.0
        echo scikit-learn^>=1.3.0
        echo scipy^>=1.11.0
        echo google-cloud-storage^>=2.10.0
        echo reportlab^>=4.0.0
        echo Pillow^>=10.0.0
    ) > requirements.txt
)

pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo   [WARNUNG] Einige Pakete konnten nicht installiert werden
    echo   Versuche einzelne Installation...
    pip install pandas numpy matplotlib seaborn scikit-learn scipy google-cloud-storage reportlab Pillow --quiet
) else (
    echo   [OK] Alle Pakete installiert
)

:: GCS-Verbindung testen
echo.
echo [3/3] GCS-Verbindung prüfen...
if exist "gcs-key.json" (
    python -c "from google.cloud import storage; print('   [OK] GCS-Verbindung erfolgreich')" 2>nul
    if errorlevel 1 (
        echo   [FEHLER] GCS-Verbindung fehlgeschlagen
    )
) else (
    echo   [FEHLER] Keine GCS-Credentials gefunden
)

echo.
echo Setup abgeschlossen!
pause
goto :MAIN_MENU

:: ==================== CUSTOM ANALYSIS ====================
:CUSTOM
cls
echo ================================================================
echo                 BENUTZERDEFINIERTE ANALYSE
echo ================================================================
echo.

set /p TICKERS="Anzahl Ticker (1-9999): "
set /p MIN_DUR="Minimale Musterdauer in Tagen (Standard 10): "
if "%MIN_DUR%"=="" set MIN_DUR=10
set /p MAX_DUR="Maximale Musterdauer in Tagen (Standard 60): "
if "%MAX_DUR%"=="" set MAX_DUR=60
set /p MAX_WIDTH="Maximale Boundary Width in %% (Standard 15): "
if "%MAX_WIDTH%"=="" set MAX_WIDTH=15

echo.
echo Konfiguration:
echo   Ticker: %TICKERS%
echo   Dauer: %MIN_DUR%-%MAX_DUR% Tage
echo   Max Width: %MAX_WIDTH%%%
echo.
set /p CONFIRM="Analyse starten? (J/N): "
if /i not "%CONFIRM%"=="J" goto :MAIN_MENU

goto :START_ANALYSIS

:: ==================== ALL DATA ====================
:ALL_DATA
cls
echo ================================================================
echo              WARNUNG: VOLLSTÄNDIGE DATENANALYSE
echo ================================================================
echo.
echo Diese Analyse wird ALLE verfügbaren Ticker analysieren!
echo Dies kann mehrere STUNDEN dauern.
echo.
set /p CONFIRM="Sind Sie sicher? (J/N): "
if /i not "%CONFIRM%"=="J" goto :MAIN_MENU

set TICKERS=9999
goto :START_ANALYSIS

:: ==================== START ANALYSIS ====================
:START_ANALYSIS
cls

:: System prüfen
if not "%READY%"=="YES" (
    echo System ist nicht bereit. Führe Setup aus...
    goto :SETUP
)

:: Python-Cache löschen für aktuelle Versionen
if exist "__pycache__" (
    rmdir /s /q __pycache__ 2>nul
    echo Python-Cache gelöscht.
)

echo ================================================================
echo                    ANALYSE LÄUFT
echo ================================================================
echo.
echo Konfiguration:
echo   Ticker: %TICKERS%
if defined MIN_DUR echo   Dauer: %MIN_DUR%-%MAX_DUR% Tage
if defined MAX_WIDTH echo   Max Width: %MAX_WIDTH%%%
echo   Datenquelle: Vollständige GCS-Historie
echo.
echo Analyse wird gestartet...
echo Dies kann einige Zeit dauern...
echo.
echo ----------------------------------------------------------------
echo.

:: Use enhanced PDF generator with detailed analysis
python enhanced_pdf_generator.py %TICKERS%

if errorlevel 1 (
    echo.
    echo ================================================================
    echo                    FEHLER AUFGETRETEN
    echo ================================================================
    echo.
    echo Bitte prüfen Sie die Fehlermeldungen oben.
    echo.
) else (
    echo.
    echo ================================================================
    echo                 ANALYSE ERFOLGREICH
    echo ================================================================
    echo.
    echo Der PDF-Report wurde generiert und sollte automatisch öffnen.
    echo.
)

:: Aufräumen
del temp_analysis.py 2>nul

:: Zurück zum Menü?
echo.
set /p AGAIN="Weitere Analyse durchführen? (J/N): "
if /i "%AGAIN%"=="J" (
    set MIN_DUR=
    set MAX_DUR=
    set MAX_WIDTH=
    goto :MAIN_MENU
)
goto :EXIT

:: ==================== HELP ====================
:HELP
cls
echo ================================================================
echo                        HILFE
echo ================================================================
echo.
echo AIv3 Pattern Analysis System - Dokumentation
echo.
echo SYSTEMANFORDERUNGEN:
echo   - Python 3.8 oder höher
echo   - GCS-Credential JSON-Datei
echo   - Internetverbindung für GCS-Zugriff
echo.
echo ANALYSEOPTIONEN:
echo   - Schnell: 25 Ticker für schnelle Ergebnisse
echo   - Standard: 50 Ticker für ausgewogene Analyse
echo   - Erweitert: 100 Ticker für umfassende Ergebnisse
echo   - Gross: 200+ Ticker für tiefgehende Analyse
echo   - ALLE: Gesamter Datensatz (sehr langsam)
echo.
echo AUSGABE:
echo   - PDF-Report mit 5 Analysen:
echo     1. Robustheits- und Sensitivitätsanalyse
echo     2. Nach-Ausbruchs-Phasenanalyse
echo     3. Regressionsanalyse
echo     4. Cluster-Analyse
echo     5. Korrelations-Heatmaps
echo.
echo GCS-PFADE:
echo   - market_data/: Primäre Marktdaten
echo   - tickers/: Zusätzliche Tickerdaten
echo.
echo MUSTER-KLASSIFIKATION:
echo   K4: Exceptional (^>75%% Gewinn)
echo   K3: Strong (35-75%% Gewinn)
echo   K2: Quality (15-35%% Gewinn)
echo   K1: Minimal (5-15%% Gewinn)
echo   K0: Stagnant (^<5%% Gewinn)
echo   K5: Failed (Breakdown)
echo.
pause
goto :MAIN_MENU

:: ==================== EXIT ====================
:EXIT
cls
echo.
echo ================================================================
echo         Vielen Dank für die Nutzung von AIv3 Analysis!
echo ================================================================
echo.
echo              Erstellt für fortgeschrittene
echo            Konsolidierungsmuster-Erkennung
echo.
echo ================================================================
echo.
timeout /t 3 >nul
exit /b 0