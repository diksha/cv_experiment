from loguru import logger

from lib.utils.py.dict.merge import DictionaryMerger


class GraphConfigBuilder(DictionaryMerger):
    """GraphConfigBuilder provides features for combining configs together"""

    def __init__(self):
        super().__init__()

    def get_config(self) -> dict:
        """
        Returns:
            dict: the combined result of all applied configs
        """
        self._log_sources()
        return self.get_merged()

    def _log_sources(self):
        """Print a debug statement explaining where each value in the config originated from"""
        lines = []
        for key, (
            defining_source,
            all_sources,
        ) in self.get_definition_sources_dict().items():
            line = f"{key} set in {defining_source}"
            if len(all_sources) > 1:
                overrides = ", ".join(all_sources[:-1])
                line += f" - overrides {overrides}"
            lines.append(line)
        logger.debug("\n".join(lines))
