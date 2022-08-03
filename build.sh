# You need pyinstaller: pip install pyinstaller
VERSION='v0.01'
pyinstaller --onefile --windowed -n "calibri-${VERSION}" --noconfirm "${PWD}/run_gui.py"