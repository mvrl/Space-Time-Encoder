#!/usr/bin/env bash
# Download the CHELSA v2.1 1981-2010 monthly climatologies used for training (11 variables x 12 months).
#   bash scripts/download_chelsa.sh /path/to/chelsa_raw
set -euo pipefail
OUT=${1:?usage: download_chelsa.sh OUT_DIR}
BASE=https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010
for var in cmi clt hurs pet pr rsds sfcWind tas tasmax tasmin vpd; do
  mkdir -p "$OUT/$var"
  for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    case $var in
      pet)  f=CHELSA_pet_penman_${mm}_1981-2010_V.2.1.tif ;;
      rsds) f=CHELSA_rsds_1981-2010_${mm}_V.2.1.tif ;;
      *)    f=CHELSA_${var}_${mm}_1981-2010_V.2.1.tif ;;
    esac
    wget -nc -q --show-progress -O "$OUT/$var/$f" "$BASE/$var/$f"
  done
done
