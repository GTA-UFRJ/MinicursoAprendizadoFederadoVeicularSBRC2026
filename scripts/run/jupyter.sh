ip=$(ip -4 addr show | grep inet | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | head -n 1)

echo "starting jupyter notebook on ip "$ip

conda activate afv_sbrc_2026
python -m jupyterlab --ip="$ip"
