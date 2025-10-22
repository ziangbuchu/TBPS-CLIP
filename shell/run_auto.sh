#!/bin/bash
# 等待直到至少有4张空闲GPU，然后将对应编号传给 train_auto.sh。
# 更稳健：通过 nvidia-smi 读取 GPU 索引与显存占用，避免依赖 gpustat 的列位置。

set -euo pipefail

# 可调参数（如需修改，可在运行前通过环境变量覆盖）
REQUIRED_GPUS=${REQUIRED_GPUS:-4}          # 需要的GPU数量
MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-4000} # 小于该显存占用(MB)视为空卡
POLL_INTERVAL=${POLL_INTERVAL:-300}        # 轮询间隔秒

mkdir -p logs

echo "[run_auto] 目标: 等待${REQUIRED_GPUS}张空闲GPU (显存<${MEM_THRESHOLD_MB}MB)。"

while true; do
  available_indices=()
  usage_report=()

  if command -v nvidia-smi >/dev/null 2>&1; then
    # 输出: "index, memory.used"（无表头、无单位）
    while IFS=, read -r idx used; do
      idx=$(echo "$idx" | xargs)
      used=$(echo "$used" | xargs)
      usage_report+=("GPU${idx}:${used}MiB")
      if [[ -n "$used" && "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt "$MEM_THRESHOLD_MB" ]; then
        available_indices+=("$idx")
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  elif command -v gpustat >/dev/null 2>&1; then
    # 备用方案：粗略从 gpustat 文本解析显存使用（不如 nvidia-smi 稳定）
    # 提示：推荐安装/使用 nvidia-smi 解析。
    line_no=0
    while IFS= read -r line; do
      # 跳过第一行时间/主机头
      if [ $line_no -eq 0 ]; then line_no=$((line_no+1)); continue; fi
      idx=$((line_no-1))
      # 提取 "xxMiB /" 中的 xx
      used=$(echo "$line" | sed -n 's/.*| *\([0-9]\+\)MiB \/.*/\1/p')
      if [ -z "$used" ]; then used=999999; fi
      usage_report+=("GPU${idx}:${used}MiB")
      if [ "$used" -lt "$MEM_THRESHOLD_MB" ]; then
        available_indices+=("$idx")
      fi
      line_no=$((line_no+1))
    done < <(gpustat)
  else
    echo "[run_auto] 未找到 nvidia-smi 或 gpustat，无法检测GPU状态。" >&2
    exit 1
  fi

  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "${ts} - 显存使用: ${usage_report[*]}"
  echo "${ts} - 可用GPU: ${available_indices[*]} (共${#available_indices[@]}张)"

  if [ ${#available_indices[@]} -ge ${REQUIRED_GPUS} ]; then
    # 选择前 REQUIRED_GPUS 张用于本次任务
    selected=("${available_indices[@]:0:${REQUIRED_GPUS}}")
    echo "[run_auto] 满足条件，使用GPU: ${selected[*]} 启动训练。"

    log_file="logs/train_$(date '+%Y%m%d_%H%M%S')_gpu-${selected[*]// /,}.log"
    # 传入具体GPU编号给 train_auto.sh（保持与现有脚本兼容，取前四个）
    nohup bash shell/train_auto.sh "${selected[0]}" "${selected[1]}" "${selected[2]}" "${selected[3]}" > "$log_file" 2>&1 & disown
    echo "[run_auto] 已启动: shell/train_auto.sh ${selected[*]}，日志: $log_file"
    break
  fi

  sleep "$POLL_INTERVAL"
done
