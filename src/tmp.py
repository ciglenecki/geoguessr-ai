import logging
import sys
from typing import TextIO

handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(level=0)
log = logging.getLogger(__name__)
# logging.StreamHandler(SocketConcatenator(sys.stdout, )
log.addHandler(handler)
log.info(f"bla bla")
log.log(10, {"test": 3})
log.error("abc")
