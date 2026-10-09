export CKPT=results/pi05-piper-vr-0913/final_model/pytorch_model.pt
export CAMERA_ORDER=observation.images.third_view,observation.images.left_wrist,observation.images.right_wrist
export PORT=5555
export PRE_RESIZE_IMAGE_SIZE=224,224

python -m deployment.model_server.server_infersystem \
    --ckpt_path "$CKPT" \
    --bind "tcp://*:${PORT}" \
    --camera_order "$CAMERA_ORDER" \
    --use_bf16 \
    --pre_resize_image_size "$PRE_RESIZE_IMAGE_SIZE" \
    --log_timing_every 20
