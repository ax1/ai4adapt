import logging
from datetime import datetime
import re


filename = "aaaaaa.log"
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, filename=filename, filemode='a', format=FORMAT)
logger = logging.getLogger('tcpserver')
logger.info('aaa')


# filename = f"AI4ADAPT_{re.sub(r'(-|:)*','',str(datetime.now().isoformat(timespec='seconds')))}.log"
# FORMAT = '%(asctime)s %(clientip)2s %(user)s %(message)s'
# logging.basicConfig(filename=filename, filemode='a', format=FORMAT, datefmt='%Y-%m-%dT%H:%M:%S')
# d = {'clientip': '192.168.0.1', 'user': 'user1'}
# logger = logging.getLogger('tcpserver')
# logger.warning('Protocol problem: %s', 'connection reset', extra=d)


# logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
# logging.warning('This will get logged to a file')
# logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# logging.warning('Admin logged out')
# logging.error("Exception occurred", exc_info=True)
