# 肥鱼按歌名学唱

基于原私聊会话投递的 AI 学唱实验插件。

## 安装与使用

解压 Release 到麦麦的 `plugins/feiyu-singing/`，目录内应直接包含 `_manifest.json` 和 `plugin.py`。

需要 Windows 本地麦麦和 OneBot，另需配套 feiyu-tools 及 FeiyuVoice 运行时（SOME、SOFA、DiffSinger、RVC、FFmpeg 等）。FEIYU_VOICE_DIR 可以指定 FeiyuVoice 目录，默认为麦麦同级目录。含源码和 worker 依赖清单。模型、参考声线、原曲、运行数据及音频均应自行合法准备。

## 许可证

插件代码采用 MIT。第三方运行时及模型采用各自许可。
