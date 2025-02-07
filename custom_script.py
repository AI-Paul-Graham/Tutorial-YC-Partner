import sys
import os
from pathlib import Path
import contextlib
import logging

def main():
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        # Suppress error message or log it silently
        sys.exit(1)

    # Get the directory of the installed package
    package_dir = Path(__file__).parent.absolute()
    app_path = package_dir / "app.py"

    if not app_path.exists():
        # Suppress error message or log it silently
        sys.exit(1)

    # Read port and address from environment variables
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    address = os.environ.get("STREAMLIT_SERVER_ADDRESS", "localhost")

    # Suppress Streamlit logging
    os.environ["STREAMLIT_SUPPRESS_LOGGING"] = "1"

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port", port,
        "--server.address", address,
        "--browser.gatherUsageStats=false",
        "--server.runOnSave=false",
        "--server.fileWatcherType=none",
        "--logger.level=error"  # Set logging level to error
    ]

    # Redirect stdout and stderr to null
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                stcli.main()
            except Exception:
                # Suppress exception traceback
                sys.exit(1)

if __name__ == '__main__':
    main()
