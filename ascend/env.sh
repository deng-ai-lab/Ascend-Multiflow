# Source in Bash. The caller chooses a compatible CANN installation explicitly.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source ascend/env.sh' >&2
    exit 1
fi
if [[ -z "${CANN_SET_ENV:-}" || ! -f "$CANN_SET_ENV" ]]; then
    printf '%s\n' 'Set CANN_SET_ENV to the set_env.sh of your compatible CANN installation.' >&2
    return 1
fi
_fidia_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_fidia_previous_cann="${ASCEND_HOME_PATH:-}"
_fidia_prune_cann() {
    local name="$1" value="${!1}" part result=""
    local -a parts
    IFS=: read -r -a parts <<< "$value"
    for part in "${parts[@]}"; do
        case "$part" in
            /usr/local/Ascend/cann*|/usr/local/Ascend/ascend-toolkit*|*/ascend-toolkit/*) continue ;;
        esac
        if [[ -n "$_fidia_previous_cann" && "$part" == "$_fidia_previous_cann"* ]]; then continue; fi
        [[ -n "$part" ]] && result="${result:+$result:}$part"
    done
    printf -v "$name" '%s' "$result"
    export "$name"
}
_fidia_prune_cann PATH
_fidia_prune_cann LD_LIBRARY_PATH
_fidia_prune_cann PYTHONPATH
unset ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_CUSTOM_OPP_PATH NPU_HOST_LIB
source "$CANN_SET_ENV"
export FIDIA_ROOT="$_fidia_root"
export PYTHONPATH="$FIDIA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TORCH_HOME="${TORCH_HOME:-$(dirname -- "$FIDIA_ROOT")/fidia-assets/torch}"
export HF_HOME="${HF_HOME:-$(dirname -- "$FIDIA_ROOT")/fidia-assets/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$(dirname -- "$FIDIA_ROOT")/fidia-assets/cache}"
unset -f _fidia_prune_cann
unset _fidia_previous_cann _fidia_root
