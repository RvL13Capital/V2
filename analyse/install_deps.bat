@echo off
echo Installing required dependencies...
echo.

pip install pandas numpy scipy
pip install google-cloud-storage google-auth
pip install matplotlib seaborn plotly
pip install scikit-learn
pip install openpyxl jinja2 markdown
pip install python-dateutil pytz schedule

echo.
echo Installation complete!
echo You can now run: python run_full_pipeline.py --test
pause