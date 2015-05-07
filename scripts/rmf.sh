#!/bin/bash
MAX_QUEUE=20
WAIT_TIME=120
MAX_TIME=04:00:00

for M in 5000
do
	for N in 100
	do
		for R in 1
		do
			for D in 0.85
			do
			for L in 10
			do
				MIN_MN=$(($M<$N?$M:$N))
				while : ; do
					[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE ]] && break
					echo "Pausing until the queue empties enough to add a new one."
					sleep $WAIT_TIME
				done
				if [ "$MIN_MN" -ge  "$R" ]; then
				JOBNAME=rsvd-$M-$N-$R-$D-$L
cat <<-EOS | sbatch
				#!/bin/bash

				#SBATCH -J $JOBNAME
				#SBATCH -o $JOBNAME.out
				#SBATCH -n 16
				#SBATCH -N 16
				#SBATCH -p gpu
				#SBATCH -t $MAX_TIME
				##SBATCH --mail-user=keith@ices.utexas.edu
				##SBATCH --mail-type=begin
				##SBATCH --mail-type=end
				#SBATCH -A PADAS

				cd ~/projects/rsvd/build/
				ibrun ./rsvd_test --m $M --n $N --r $R --d $D --l $L

				exit 0
				EOS
			fi
			done
		done
	done
done
done
