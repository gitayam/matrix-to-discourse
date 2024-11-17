from maubot import Plugin

class MatrixToDiscourseBot(Plugin):
    async def start(self) -> None:
        self.log.info("Plugin loaded successfully!")