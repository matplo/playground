#!/bin/bash

#!/bin/bash

# Define an array of pairs
pairs=("40 60" "60 80" "80 100" "100 120")

jetR=0.2

# Loop over the array
for pair in "${pairs[@]}"; do
    # Split the pair into two variables
    IFS=' ' read -ra pts <<< "$pair"
    pt1=${pts[0]}
    pt2=${pts[1]}
    # Use the ptues
    echo "pt 1: $pt1, pt 2: $pt2"
	#./draw_mass_jewel.py --jetptmin ${pt1} --jetptmax ${pt2} ./output_v2.4.0/ptmin${pt1} --jetR ${jetR} jm_vac_${pt1}_${jetR}.root out_lhcvac
	#./draw_mass_jewel.py --jetptmin ${pt1} --jetptmax ${pt2} ./output_v2.4.0/ptmin${pt1} --jetR ${jetR} jm_med_${pt1}_${jetR}.root out_lhc10cent
	./draw_mass_jewel.py --jetptmin ${pt1} --jetptmax ${pt2} ./output_v2.4.0/ptmin${pt1} --jetR ${jetR} --bindata alice_data.root:hmain_mass_R0.2_${pt1}-${pt2}_trunc jm_vac_${pt1}_${jetR}.root out_lhcvac
	./draw_mass_jewel.py --jetptmin ${pt1} --jetptmax ${pt2} ./output_v2.4.0/ptmin${pt1} --jetR ${jetR} --bindata alice_data.root:hmain_mass_R0.2_${pt1}-${pt2}_trunc jm_med_${pt1}_${jetR}.root out_lhc10cent
done

