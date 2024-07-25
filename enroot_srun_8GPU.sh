which python
source /root/miniconda3/bin/activate
cd /workspace/BenchmarkSP/

LOG_DIR=.
NSYS_ITER=10 # -1: off, >0 to enable recommend: 10
NSYS_ITER_RANGE=2

if (( $NSYS_ITER >= 0 )); then
    mkdir -p ${LOG_DIR}/nsys_reports
    NSYS_CMD="/workspace/target-linux-x64/nsys profile --force-overwrite true -o ${LOG_DIR}/nsys_reports/ours-32GPU-$NODE_RANK --capture-range=cudaProfilerApi"
    NSYS_ARGS="
        --profile --profile-step-start $NSYS_ITER --profile-step-end $(($NSYS_ITER + $NSYS_ITER_RANGE))
    "
else
    NSYS_CMD=""
    NSYS_ARGS=""
fi

$NSYS_CMD bash enroot_srun_single_8GPU.sh ${NSYS_ARGS}
