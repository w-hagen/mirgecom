#!/bin/bash

set -o nounset -o errexit
TOPDIR=${1}
[ -z "${CEESD_CONDA_KEY}" ] && CEESD_CONDA_KEY="dgfem"
[ -z "${TOPDIR}" ] && TOPDIR="."
source ${TOPDIR}/parse_requirements.sh

parse_requirements ${TOPDIR}/requirements.txt

echo "==== Installing pip packages"
# Required for pyopencl
python -m pip install pybind11 mako
MY_CONDA_PATH="$(conda info --envs | grep ${CEESD_CONDA_KEY} | awk '{print $3}')"
printf "Using CONDA ENV PATH: ${MY_CONDA_PATH}\n"

for i in "${!module_names[@]}"; do
    name=${module_names[$i]}
    branch=${module_branches[$i]}
    url=${module_urls[$i]}

    if [[ -z $url ]]; then
        echo "=== Installing non-git module $name"
        pip install --upgrade $name
    else
        echo "=== Installing git module $name $url ${branch/--branch /}"
        [[ ! -d $name ]] && git clone --recursive $branch $url

        [[ $name == "pyopencl" ]] && (cd $name && ./configure.py --cl-inc-dir=$MY_CONDA_PATH/include --cl-lib-dir=$MY_CONDA_PATH/lib --ldflags="" --cl-libname=OpenCL)

        (cd $name && pip install -v -e .)
    fi
done

pip install -v -e ${TOPDIR}
