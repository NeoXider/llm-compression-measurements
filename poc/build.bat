@echo off
rem Build the PoC CUDA programs (CUDA 11.8, MSVC 14.38). ARCH default sm_86 (RTX 30xx); e.g. set ARCH=sm_90 for H100/H200, sm_80 for A100.
if "%ARCH%"=="" set ARCH=sm_86
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 10.0.22621.0 >nul 2>&1
if errorlevel 1 exit /b 1
cd /d "%~dp0"
if not exist bin mkdir bin
%NVCC% -O3 -arch=%ARCH% -std=c++17 -o bin\kvread_bench.exe kvread_bench.cu
if errorlevel 1 exit /b 1
%NVCC% -O3 -arch=%ARCH% -std=c++17 -o bin\kvread_base_plus.exe kvread_base_plus.cu
if errorlevel 1 exit /b 1
%NVCC% -O3 -arch=%ARCH% -std=c++17 -o bin\bf16_gemv_bench.exe bf16_gemv_bench.cu
if errorlevel 1 exit /b 1
%NVCC% -O3 -arch=%ARCH% -std=c++17 -o bin\lazy_pool.exe lazy_pool.cu -lcuda -ldxgi
exit /b %errorlevel%
