python3 lbpmslim-launcher.py

running pip install for these three dependencies should be sufficient to run the launcher. Let me know if it doesn't work.

pyqt5
numpy
connected-components-3d


You'll need to overwrite the install directories in the runLBPMSinglePhase and runLBPMTwoPhase scripts before running.
The two variables are initialised right at the start of the function, simply change them to the location of your own LBPM-slim install.