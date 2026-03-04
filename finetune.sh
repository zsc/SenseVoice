# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
gpu_num=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically
default_model_name_or_model_dir="iic/SenseVoiceSmall"
local_model_dir="${workspace}/modelscope_models/iic/SenseVoiceSmall"
if [ -d "${local_model_dir}" ]; then
    default_model_name_or_model_dir="${local_model_dir}"
fi
model_name_or_model_dir=${MODEL_NAME_OR_MODEL_DIR:-${default_model_name_or_model_dir}}
remote_code=${REMOTE_CODE:-"${workspace}/model.py"}

## option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}


# data dir, which contains: train.json, val.json
train_data=${TRAIN_JSONL:-${workspace}/data/train_example.jsonl}
val_data=${VAL_JSONL:-${workspace}/data/val_example.jsonl}

# train knobs (override via env)
batch_size=${BATCH_SIZE:-2000}
num_workers=${NUM_WORKERS:-4}
max_epoch=${MAX_EPOCH:-10}
lr=${LR:-0.00002}
use_fp16=${USE_FP16:-false}
accum_grad=${ACCUM_GRAD:-1}
log_interval=${LOG_INTERVAL:-1}
resume=${RESUME:-true}
interval=${INTERVAL:-20000}
keep_nbest_models=${KEEP_NBEST_MODELS:-5}
avg_nbest_model=${AVG_NBEST_MODEL:-5}

# exp output dir
output_dir=${OUTPUT_DIR:-"./outputs"}
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/deepspeed_conf/ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

echo $DISTRIBUTED_ARGS

# funasr trainer path
train_tool=$(python -c "import funasr.bin.train_ds as m; print(m.__file__)" 2>/dev/null) || true
if [ -z "${train_tool}" ] || [ ! -f "${train_tool}" ]; then
    if command -v funasr-train-ds >/dev/null 2>&1; then
        train_tool=$(command -v funasr-train-ds)
    else
        echo "Error: FunASR train tool not found (need funasr.bin.train_ds or funasr-train-ds)."
        exit 1
    fi
fi
echo "Using funasr trainer: ${train_tool}"

torchrun $DISTRIBUTED_ARGS \
${train_tool} \
++model="${model_name_or_model_dir}" \
++trust_remote_code=true \
++remote_code="${remote_code}" \
++disable_update=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=${batch_size}  \
	++dataset_conf.sort_size=1024 \
	++dataset_conf.batch_type="token" \
	++dataset_conf.num_workers=${num_workers} \
	++train_conf.max_epoch=${max_epoch} \
	++train_conf.log_interval=${log_interval} \
	++train_conf.resume=${resume} \
	++train_conf.validate_interval=${interval} \
	++train_conf.save_checkpoint_interval=${interval} \
	++train_conf.keep_nbest_models=${keep_nbest_models} \
	++train_conf.avg_nbest_model=${avg_nbest_model} \
	++train_conf.use_deepspeed=false \
	++train_conf.deepspeed_config=${deepspeed_config} \
	++train_conf.use_fp16=${use_fp16} \
	++train_conf.accum_grad=${accum_grad} \
	++optim_conf.lr=${lr} \
++output_dir="${output_dir}" &> ${log_file}
