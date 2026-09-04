import asyncio
from pathlib import Path
from dotenv import load_dotenv
from src.bot.adapter_registry import enabled_adapters, validate_enabled_adapters
from src.bot.adapter_support import AdapterRuntime
from src.bot.process_lock import BotProcessLock
from src.memory.maintenance import MemoryMaintenanceConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def main():
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    specs = enabled_adapters()
    validate_enabled_adapters(specs)
    starters = [(spec, spec.load_starter()) for spec in specs]
    logger.info("已通过适配器预检：%s", ", ".join(spec.name for spec in specs))

    runtime = await AdapterRuntime.create()

    async def run_adapter(name: str, starter) -> None:
        logger.info("正在启动 %s adapter", name)
        await starter(runtime)
        raise RuntimeError(f"{name} adapter stopped unexpectedly")

    try:
        async with asyncio.TaskGroup() as group:
            for spec, starter in starters:
                group.create_task(
                    run_adapter(spec.name, starter),
                    name=f"adapter:{spec.name}",
                )
    finally:
        await runtime.close()

if __name__ == "__main__":
    try:
        project_root = Path(__file__).resolve().parent
        load_dotenv(project_root / ".env")
        maintenance = MemoryMaintenanceConfig.from_project(project_root)
        with BotProcessLock(maintenance.bot_lock_path):
            asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭机器人...")
