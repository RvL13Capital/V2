TRANS Pipeline Shortcuts
========================

Double-click any .bat file to run the task.

Available Shortcuts:
--------------------

TRAINING:
  Train_EU.bat              - Train European market model (asks for epochs)
  Train_US.bat              - Train US market model (asks for epochs)

SEQUENCE GENERATION:
  Generate_EU_Sequences.bat - Generate sequences from EU patterns
  Generate_US_Sequences.bat - Generate sequences from US patterns

FULL PIPELINES:
  Full_EU_Pipeline.bat      - Complete EU pipeline (detect + generate + train)

UTILITIES:
  Run_Tests.bat             - Run the test suite
  View_Task_History.bat     - See recent task results

MAIN DASHBOARD:
  ../Dashboard.bat          - Interactive menu-driven dashboard (RECOMMENDED)

Tips:
-----
- All tasks run through the external task runner
- Output is saved to output/tasks/<timestamp>_<name>/
- Check task history to see results
- Use the Dashboard.bat for the best experience
