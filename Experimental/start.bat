@echo off
echo ============================================================
echo    Heart Disease Prediction System
echo ============================================================
echo.
echo Starting the application...
echo.
echo Make sure you have Python installed and dependencies installed
echo If not, run: pip install -r requirements.txt
echo.
echo The application will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python run.py --init-data

pause
