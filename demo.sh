python demo.py \
    --config configs/config_ude.yaml \
    --eval_folder results/eval \
    --eval_name ude \
    --repeat_times 1 \
    --repeat_times_dmd 3 \
    --add_noise False \
    --use_dmd True \
    --condition_type audio \
    --condition_input sample_data/a2m

python visualization/visualize_mesh.py \
    --input_folder results/eval/ude/output/a2m \
    --output_folder results/eval/ude/animation/a2m \
    --fps 30