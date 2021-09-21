import requests
class RouteNotFound(Exception):
    def __init__(self, request: requests.Request):
        self.request = request
        super().__init__("Route for request not found: '%s'" % request.url)
