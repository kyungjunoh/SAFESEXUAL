SD_MODEL_ID=v1-4
CONFIG_PATH="./configs/sd_config.json"
ERASE_ID=std
MODEL_ID="CompVis/stable-diffusion-v1-4"
ATTACK_TYPE=p4d
attack_data="./datasets/nudity.csv"
thr=0.6

# Sweep the data-driven P_c rank (and optionally the corpus purity threshold).
# Higher rank captures more nudity directions (lower ASR) but risks erasing benign
# content (quality loss) -> watch COCO FID/CLIP alongside ASR when picking the knee.
PC_THR=30
for PC_RANK in 5 10 20 40
do
    save_dir="./results/gen_SAFREEpcdd_SD${SD_MODEL_ID}_${ATTACK_TYPE}_thr${PC_THR}_rank${PC_RANK}/"
    echo "=== pc_rank=${PC_RANK} pc_nudity_thr=${PC_THR} -> ${save_dir} ==="
    python generate_safree.py \
        --config $CONFIG_PATH \
        --data ${attack_data} \
        --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
        --category nudity \
        --num-samples 1 \
        --erase-id $ERASE_ID \
        --model_id $MODEL_ID \
        --nudity_thr $thr \
        --device cuda:0 \
        --save-dir ${save_dir} \
        --safree -svf -lra \
        --pc_corpus ./datasets/i2p.csv \
        --pc_nudity_thr ${PC_THR} \
        --pc_rank ${PC_RANK}
done
