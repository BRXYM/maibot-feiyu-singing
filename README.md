# 肥鱼按歌名学唱

基于原私聊会话投递的 AI 学唱实验插件。

## 安装与使用

解压 Release 到麦麦的 `plugins/feiyu-singing/`，目录内应直接包含 `_manifest.json` 和 `plugin.py`。

需要 Windows 本地麦麦和 OneBot，另需配套 feiyu-tools 及 FeiyuVoice 运行时（SOME、SOFA、DiffSinger、RVC、FFmpeg 等）。FEIYU_VOICE_DIR 可以指定 FeiyuVoice 目录，默认为麦麦同级目录。含源码和 worker 依赖清单。模型、参考声线、原曲、运行数据及音频均应自行合法准备。

整首模式在 GPU 上分轨、分段换声，再混入伴奏。工作进程会复核孤立的短促跳音，动态歌词视频保留开头人声。发送、状态查询和失败说明绑定原私聊。

需要支持 W61 v2 的 FeiyuVoice 运行时与模型清单。在配套 WSL 服务的 server 目录中使用 git apply --check 检查 runtime-adapter/singing-rvc.patch，匹配后用 git apply 应用，并将 runtime-adapter/pitch_review.py 复制到 feiyu_runtime/。已有 W61 修改的部署需按补丁合并。voice-slot.sh 切换配套的 feiyu-tts-resident 和 feiyu-voice-api systemd 服务。FEIYU_MAIBOT_DIR 可指定宿主目录，代点好友功能读取 FEIYU_OWNER_QQ 与 FEIYU_BOT_QQ。

## 许可证

插件代码采用 MIT。第三方运行时及模型采用各自许可。
