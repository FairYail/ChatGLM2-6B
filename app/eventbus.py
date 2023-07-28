from app.config import Config
from base_log import llog
from etcd3 import Etcd3Client
import threading
from etcd3.events import PutEvent, DeleteEvent


# EtcdWatcherThread 事件监听
class EtcdWatcherThread(threading.Thread):
    _initialized = False
    etcd_client = None

    def __new__(cls, *args, **kwargs):
        if not cls._initialized:
            cls.initEventBus()
            cls._initialized = True
        return super().__new__(cls)

    def __init__(self, key, callback):
        super().__init__()  # 调用父类的初始化方法
        self.daemon = True
        self.key = key
        self.callback = callback

    # etcd 事件监听初始化
    @classmethod
    def initEventBus(cls):
        host = Config().get_config_info("EtcdEventBus", "url")
        port = Config().get_config_info("EtcdEventBus", "port")
        dialTimeout = float(Config().get_config_info("EtcdEventBus", "dialTimeout"))
        try:
            conEtcd = Etcd3Client(host=host, port=port, timeout=dialTimeout)
            # 尝试执行任意一个 etcd3 方法来检查连接
            conEtcd.get('/ping')
        except Exception as e:
            llog.error("EtcdEventBus init failed... host:{}，port:{}".format(host, port))
            raise Exception(f"EtcdEventBus init failed: {e}")

        llog.info("EtcdEventBus init success... host:{}，port:{}".format(host, port))
        cls.etcd_client = conEtcd

    def run(self):
        watch_response, cancel = self.etcd_client.watch_prefix(self.key)
        for event in watch_response:
            key = event.key.decode('utf-8')
            value = event.value.decode('utf-8')
            if isinstance(event, PutEvent):
                self.callback(key, value, "PUT")
            elif isinstance(event, DeleteEvent):
                self.callback(key, value, "DELETE")
            else:
                llog.info("未定义事件类型{}".format(str(event)))

    # 发布消息
    @classmethod
    def publish_message(cls, key=str, value=str):
        try:
            cls.etcd_client.put(key, value)
            llog.info("Published message: key={}, value={}".format(key, value))
        except Exception as e:
            llog.error("Failed to publish message: key={}, value={}".format(key, value))
            raise Exception(f"Failed to publish message: {e}")


# 注册事件监听
def RegisterEtcdWatcher(key: str, callback):
    llog.info(f"监听开启: key:{key}")
    watcher_thread = EtcdWatcherThread(key, callback)
    watcher_thread.start()
