# Script to generate dataset for training

echo "Sampling videos for clips..."
python utils/clip_parser.py $AUTOHL_SOURCE $AUTOHL_CLIPS
echo "Clip generation completed."

echo "Splitting files..."
python train_test_split.py $AUTOHL_CLIPS $AUTOHL_SPLIT
echo "Train/val/test split completed."
