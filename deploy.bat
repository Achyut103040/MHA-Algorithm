@echo off
echo ========================================
echo MHA Toolbox - PyPI Deployment
echo ========================================
echo.

echo [1/3] Uploading to PyPI...
python -m twine upload dist/*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Upload failed!
    echo Check DEPLOY_NOW.md for troubleshooting
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Package uploaded to PyPI!
echo ========================================
echo.
echo Your package is now available at:
echo https://pypi.org/project/mha-toolbox/
echo.
echo Anyone can now install it with:
echo pip install mha-toolbox
echo.
echo [2/3] Next steps:
echo   - Test installation: pip install mha-toolbox
echo   - Update GitHub: git push origin main
echo   - Create GitHub release
echo   - Share on social media
echo.
echo See DEPLOY_NOW.md for detailed instructions
echo.
pause
