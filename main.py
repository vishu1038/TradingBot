import tkinter as tk
import logging

from connectors.binance_futures import BinanceFuturesClient

logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")

stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

if __name__ == "__main__":
    binance = BinanceFuturesClient("5727e21465d0a34be5e3f82bfc1882eca641dc031647c2f33ac34e788d4daf02",
                                   "a27bd67c08753f4948b27990fd591b74c3a415fee8b811bddb4ffa4563c9aa37", True)
    print(binance.get_balances())
    print(binance.place_order("BTCUSDT","BUY", 0.01, "LIMIT", 20000, "GTC"))
    print(binance.get_order_status("BTCUSDT", 2977862718))
    print(binance.cancel_order("BTCUSDT", 2977862718))

    root = tk.Tk()
    root.mainloop()
