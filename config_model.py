from maibot_sdk import Field, PluginConfigBase

class PluginSection(PluginConfigBase):
    enabled: bool = Field(default=True, description='启用按歌名学唱')
    config_version: str = Field(default='1.0.0', description='配置版本')
    automatic_delivery: bool = Field(default=True, description='合成和自动检查通过后，向发起任务的原私聊发送一次试唱')
    max_pending_jobs: int = Field(default=2, ge=1, le=3, description='等待和运行中的任务上限；实际推理串行执行')
    default_seconds: int = Field(default=30, ge=5, le=45, description='默认学唱时长上限，秒；优先连续2–3句')

class SingingConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
