
#!/bin/bash

### SET VARIABLES ONLY HERE
run_dir="path/to/run"
output_path="${run_dir}/stylos_txt"

########## BELOW AUTOMATICALLY
# first generate test renders
python evaluation/generate_test_images.py \
--run_dir "$run_dir" \
--content_path path/to/stylos_evaluation_data/content \
--style_path path/to/stylos_evaluation_data/style_descriptions.txt \
--output_path "$output_path"

# art-score, consistency from stylos
python evaluation/eval_artscore_consistency_from_stylos.py \
--base_dir $output_path/generated \
--artscore_model path/to/loss@listMLE_model@resnet50_denseLayer@True_batch_size@16_lr@0.0001_dropout@0.5_E_8.pth \
--model path/to/raft-things.pth


for scene in Garden Truck Train M60
do
  echo "Processing scene: $scene"

# Art-FID
  python evaluation/eval_artfid.py \
  --sty "$output_path/style/$scene" \
  --cnt "$output_path/content/$scene" \
  --tar "$output_path/generated/$scene" \
  --output_path "$output_path/results_artfid_$scene.txt"

  # Histogram loss
  python evaluation/eval_histogan.py \
    --sty "$output_path/style/$scene" \
    --tar "$output_path/generated/$scene" \
    --output_path "$output_path/results_cmloss_$scene.txt"

done