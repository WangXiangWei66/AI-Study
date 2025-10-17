@echo off
setlocal EnableDelayedExpansion
del chinaz.com.txt /s
del 说明.htm /s
del 西西软件园.txt /s
del 西西软件.url /s
::copy d:\jb51tools\jb_down\soft\ !cd!
mkdir jb51.net
del %0 | move *.* jb51.net