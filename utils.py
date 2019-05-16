from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import logging

log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


def setupLog():
    """
    Reset logging: to make the logging.basicConfig() work again.

    Step 1: remove all the root.handlers
       logging.basicConfig() does nothing if the root logger already has handlers
            configured for it.
    Step 2: setup logging format
        logging.basicConfig() will create and add a default handler to the root logger.
    """
    if logging.root:
        logging.root.handlers = []
    fmt = "%(levelname).3s[%(asctime)s %(filename)s:%(lineno)d] %(message)s"
    dtfmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=dtfmt)
    return

