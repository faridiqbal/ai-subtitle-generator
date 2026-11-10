@echo off
REM SRT Translator launcher (Windows). Shares your user folder with the container.
REM Use another folder:  set HOST_ROOT=D:\Movies  then run.bat
REM Change settings:     run.bat --setup
REM Rebuild the image:   set REBUILD=1  then run.bat
setlocal
cd /d "%~dp0"
if "%HOST_ROOT%"=="" set "HOST_ROOT=%USERPROFILE%"
set IMAGE=srt-translator
if not "%REBUILD%"=="" goto build
docker image inspect %IMAGE% >nul 2>&1 && goto run
:build
echo Building Docker image %IMAGE% (first time only)...
docker build -t %IMAGE% . || exit /b 1
:run
docker run -it --rm -v "%HOST_ROOT%:/host" -e "HOST_ROOT=%HOST_ROOT%" -v srt-config:/config -v whisper-models:/models -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e GEMINI_API_KEY %IMAGE% %*
