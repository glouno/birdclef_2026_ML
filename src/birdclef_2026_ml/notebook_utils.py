# src/startup.py
import os
import sys
import threading
from pathlib import Path


_INIT_RETRY_LIMIT = 25
_init_retry_count = 0


def init_notebook(seaborn_theme="darkgrid"):
    """Initialize notebook environment"""
    global _init_retry_count

    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return

    if ip is None:
        # sitecustomize can run before the IPython shell is fully constructed.
        if _init_retry_count < _INIT_RETRY_LIMIT and any(
            marker in os.environ
            for marker in ("JPY_PARENT_PID", "PYKERNEL_LAUNCHER", "VSCODE_INJECTION")
        ):
            _init_retry_count += 1
            timer = threading.Timer(0.2, init_notebook, kwargs={"seaborn_theme": seaborn_theme})
            timer.daemon = True
            timer.start()
        return

    _init_retry_count = _INIT_RETRY_LIMIT

    try:
        import seaborn as sns
        sns.set_theme(style=seaborn_theme)
    except Exception:
        pass

    def _enable_autoreload() -> bool:
        try:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
            return True
        except Exception:
            return False

    if _enable_autoreload():
        return

    # During kernel bootstrap, magics can be unavailable briefly.
    events = getattr(ip, "events", None)
    if events is None:
        return

    def _late_enable(*_args, **_kwargs):
        if _enable_autoreload():
            try:
                events.unregister("pre_run_cell", _late_enable)
            except Exception:
                pass

    try:
        events.register("pre_run_cell", _late_enable)
    except Exception:
        pass


def run_jupyter():
    """Run Jupyter with project context"""
    import os
    import subprocess

    project_root = Path(__file__).parent.parent

    # Set environment
    env = os.environ.copy()
    env['JUPYTER_CONFIG_DIR'] = str(project_root / '.jupyter')
    env['PYTHONPATH'] = str(project_root / 'src') + ':' + env.get('PYTHONPATH', '')

    # Run Jupyter
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'notebook',
        str(project_root / 'notebooks')
    ], env=env)


if __name__ == '__main__':
    run_jupyter()
