python -m paddle.distributed.launch eval_paddle.py --model_name swinl_at --attack_types autoattack > logs/swinl_at_autoattack.txt
sleep 30
python -m paddle.distributed.launch eval_paddle.py --model_name xcitl_at --attack_types autoattack > logs/xcitl_sota_autoattack.txt