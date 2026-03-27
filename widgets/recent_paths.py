"""Recent file paths manager and menu builder.

Persists a bounded list of recently-opened file paths using QSettings,
and provides a helper to populate a QMenu with those paths.
"""

from pathlib import Path
from typing import List
from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

class RecentPaths:
    """Manage a most-recently-used list of file paths via QSettings.

    Paths are stored in order of last access, with the most recent first,
    and the list is automatically trimmed to *limit* entries.
    """

    def __init__(self, org: str, app: str, key: str = "recent_paths", limit: int = 10):
        """Initialise the recent-paths store.

        Args:
            org: Organisation name for QSettings.
            app: Application name for QSettings.
            key: Settings key under which the list is stored.
            limit: Maximum number of paths to keep.
        """
        self.settings = QSettings(org, app)
        self.key = key
        self.limit = limit

    def add(self, path: str) -> None:
        """Add *path* to the top of the recent list, deduplicating if present."""
        p = str(Path(path).resolve())
        items = [x for x in self.list() if x != p]
        items.insert(0, p)
        self.settings.setValue(self.key, items[: self.limit])

    def list(self) -> List[str]:
        """Return the stored paths, most-recent first."""
        val = self.settings.value(self.key, [], list)
        return [str(x) for x in val]

    def clear(self) -> None:
        """Remove all stored recent paths."""
        self.settings.remove(self.key)

def populate_recent_menu(menu: QMenu, recent: RecentPaths, on_open):
    """Rebuild a QMenu with the recent file paths.

    Each entry triggers *on_open(path)* when clicked.  A "Clear list"
    action is appended at the bottom.

    Args:
        menu: The QMenu to populate.
        recent: RecentPaths instance supplying the path list.
        on_open: Callback invoked with the chosen path string.
    """
    menu.clear()
    paths = recent.list()
    if not paths:
        a = QAction("(No recent items)", menu); a.setEnabled(False); menu.addAction(a)
        return
    for p in paths:
        act = QAction(p, menu)
        act.triggered.connect(lambda _, path=p: on_open(path))
        menu.addAction(act)
    menu.addSeparator()
    clear_act = QAction("Clear list", menu)
    clear_act.triggered.connect(recent.clear)
    clear_act.triggered.connect(lambda: populate_recent_menu(menu, recent, on_open))
    menu.addAction(clear_act)

