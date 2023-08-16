#!/bin/bash

i=$1
if [ -z $i ]
then
  echo "No ripper id provided."
  exit 1
fi

set -euo pipefail
export PYTHONPATH=.

user=kevingd
if [ $i -ge 8 ]
then
  user=kgd
fi

host=ripper$i
base=$user@$host:data/amaze

info=""
# info=--info=progress2
rsync -avzh $info $base/ remote/dd -f '+ */' -f '+ *.png' -f '+ *.zip' -f '+ *.csv' -f '+ events.out.*' -f '+ log' -f '- *'

# [ -z ${VIRTUAL_ENV+x} ] && source ~/work/code/vu/venv/bin/activate
# ./bin/tools/retina_summary.py
#
# groups=$(ls -d remote/collect_v3/*/ | cut -d '-' -f 2 | sort -u)
# echo $groups
# cd remote/collect_v3
# sorted(){
#   find *-$1-100K -name 'best.json' \
#   | xargs jq -r '"\(input_filename) \(.fitnesses.collect)"' \
#   | sort -k2gr \
#   | awk -F. -ve=$2 '{print $1"."e}'
# }
# for g in $groups
# do
#   if ls *-$g-*/best.trajectory.png >/dev/null 2>&1
#   then
#     montage -geometry +0+0 -label '%d' $(sorted $g trajectory.png) $g.trajectories.png
#   fi
#
#   if ls *-$g-*/best.cppn.png >/dev/null 2>&1
#   then
#     montage -geometry '256x256>+0+0' -label '%d' $(sorted $g cppn.png) $g.cppn.png
#   fi
#
#   for t in weight leo bias
#   do
#     if ls *-$g-*/best.cppn.$t.png >/dev/null 2>&1
#     then
#       montage -geometry '+10+10' -label '%d' $(sorted $g cppn.$t.png) $g.cppn.$t.png
#     fi
#   done
# done
# cd -
#
# pgrep -a feh | grep 'collect_v3/$' > /dev/null || feh -.Z --reload 10 remote/collect_v3/ 2>/dev/null &
