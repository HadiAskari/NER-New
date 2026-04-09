#python -m src.serve.cli_ser_combined_security     --model_path src/train/saved_models/hardware_full  --tensor_parallel_size 1     --max_input_length 2048

#CUDA_VISIBLE_DEVICES=1,2,


python -m src.serve.cli_ser_for_demo \
--text_new "Software vulnerabilities, like buffer overflow and SQL injection flaw, can expose systems to severe security risks, allowing attackers to gain unauthorized access or execute malicious code." \
--model_path src/train/saved_models/vulnerability_full  \
--tensor_parallel_size 1 \
--max_input_length 2048 \
--max_new_tokens 2048
