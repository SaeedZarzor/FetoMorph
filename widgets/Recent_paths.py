from deps import *

class RecentPaths:
    def __init__(self, org: str, app: str, key: str = "recent_paths", limit: int = 10):
        self.settings = QSettings(org, app)
        self.key = key
        self.limit = limit

    def add(self, path: str) -> None:
        p = str(Path(path).resolve())
        items = [x for x in self.list() if x != p]
        items.insert(0, p)
        self.settings.setValue(self.key, items[: self.limit])

    def list(self) -> List[str]:
        val = self.settings.value(self.key, [], list)
        return [str(x) for x in val]

    def clear(self) -> None:
        self.settings.remove(self.key)

def populate_recent_menu(menu: QMenu, recent: RecentPaths, on_open):
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

