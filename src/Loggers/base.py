import logging

class BaseLogger:
    def __init__(self, log_file):
        # Настройка базового логгера
        logging.basicConfig(
            filename=log_file,  # Логирование в файл
            level=logging.INFO,  # Уровень логирования
            format="%(asctime)s - %(levelname)s - %(message)s",  # Формат логов
            datefmt="%Y-%m-%d %H:%M:%S"  # Формат даты
        )

        # Добавляем обработчик для вывода в консоль
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Уровень логирования для консоли
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Добавляем обработчик консоли к глобальному логгеру
        logging.getLogger().addHandler(console_handler)

    def log(self, message, level="info"):
        """
        Логирует сообщение с указанным уровнем.
        :param message: Сообщение для логирования.
        :param level: Уровень логирования ("info", "debug", "warning", "error", "critical").
        """
        if level == "debug":
            logging.debug(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)
        elif level == "critical":
            logging.critical(message)
        else:
            logging.info(message)
