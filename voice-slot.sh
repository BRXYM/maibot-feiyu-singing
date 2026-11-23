#!/usr/bin/env bash
set -eu
# The pipeline owns stdin. EOF also releases the models if its Windows process exits.
resident=feiyu-tts-resident.service
api=feiyu-voice-api.service
restore=()
for unit in "$resident" "$api"; do
    if systemctl is-active --quiet "$unit"; then restore+=("$unit"); fi
done
restore_voice() {
    if (( ${#restore[@]} )); then systemctl start "${restore[@]}"; fi
}
trap restore_voice EXIT
trap 'exit 1' HUP INT TERM
if (( ${#restore[@]} )); then systemctl stop "$api" "$resident"; fi
echo READY
read -r _ || true
