NGPU=1 CONFIG_FILE="./llama3_1b.toml" TRAIN_FILE="torchtitan_train" bash /workspace/main/torchtitan/run_train.sh
NGPU=1 CONFIG_FILE="./llama3_1b_fp8.toml" TRAIN_FILE="torchtitan_train" bash /workspace/main/torchtitan/run_train.sh
