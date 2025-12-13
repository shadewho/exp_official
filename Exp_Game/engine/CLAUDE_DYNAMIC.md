# Dynamic Mesh System - Development Guide

## Development Workflow

**ALWAYS use the dev logger to debug and test results.**
read /developer/CLAUDE_LOGGER.md
always follow the propper output flow
logs go to the log output file: "C:\Users\spenc\Desktop\engine_output_files\diagnostics_latest.txt"
never log to the python console in blender. too much overhead




## Goal:
-I want as much unity in the static and dynamic mesh physics system as possible

-I dont want to maintain seperate physics or view logic for each mesh type

-I want fast and efficient dynamic code that shares as much of the static 
mesh system as possible

-Its so essential if we ever have a problem that we see the results in the 
logger. The character visualizer should also behave the same for static and proxy meshes and 
change color or visuals the same

-Performance should always be most important in all solutions, this is a game engine that
needs to work on many computers and systems of varying capabilities


**End of Document**
