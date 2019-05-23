# To avoid import error on python3
class Connection:
    def __init__(self, base_url, username=None, password=None, timeout=None):
        self.base_url = base_url
        self.username = username
        self.timeout = timeout


    def request_get(self, resource, args = None, headers={}):
        pass

    def request_post(self, resource, args=None, body=None, filename=None, headers={}):
        pass