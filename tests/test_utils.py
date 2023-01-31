import re

import pipeline.util


def test_package_version():
    version = pipeline.util.package_version()
    # We don't care about the particular version, but
    # it should match a particular form.
    assert re.match(r"[0-9]{1,}\.[0-9]{1,}\.[0-9]{1,}", version)
