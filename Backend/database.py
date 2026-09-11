
class _Result:
    def __init__(self, matched_count=0):
        self.matched_count = matched_count


class InMemoryCollection:
    def __init__(self):
        self._docs = {}

    def find_one(self, query: dict):
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    def insert_one(self, doc: dict):
        self._docs[doc["_id"]] = doc
        return doc

    def update_one(self, query: dict, update: dict):
        doc = self.find_one(query)
        if doc:
            for k, v in update.get("$set", {}).items():
                doc[k] = v
            return _Result(matched_count=1)
        return _Result(matched_count=0)

    def find(self, query: dict = None):
        query = query or {}
        matches = [d for d in self._docs.values() if all(d.get(k) == v for k, v in query.items())]
        return _Cursor(matches)

    def create_index(self, *args, **kwargs):
        # No-op — indexes don't matter for an in-memory dict.
        pass


class _Cursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, field, direction=1):
        self._docs.sort(key=lambda d: d.get(field), reverse=(direction == -1))
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class _InMemoryDB:
    def __init__(self):
        self.personnel = InMemoryCollection()
        self.assessment_sessions = InMemoryCollection()
        self.welfare_interventions = InMemoryCollection()


db = _InMemoryDB()


def get_db():
    return db