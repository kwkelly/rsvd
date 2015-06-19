#!/bin/bash
MAX_QUEUE=20
WAIT_TIME=120
MAX_TIME=00:10:00
TOL=0.00000001

for M in 1000
do
	for N in 500
	do
		for R in 25
		do
			for D in 0.90
			do
			for L in 10
			do
			for NODES in 1 2 3 4 6 8
			do
				for Q in 0
				do
				MIN_MN=$(($M<$N?$M:$N))
				while : ; do
					[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE ]] && break
					echo "Pausing until the queue empties enough to add a new one."
					sleep $WAIT_TIME
				done
				if [ "$MIN_MN" -ge  "$R" ]; then
				#JOBNAME=rsvd-$M-$N-$R-$D-$L
				#JOBNAME=test100
				JOBNAME=tests-$NODES-$M-$N-$R-$D
cat <<-EOS | sbatch
				#!/bin/bash

				#SBATCH -J $JOBNAME
				#SBATCH -o $JOBNAME.out
				#SBATCH -n $((NODES*8))
				#SBATCH -N $NODES
				#SBATCH -p gpu
				#SBATCH -t $MAX_TIME
				##SBATCH --mail-user=keith@ices.utexas.edu
				##SBATCH --mail-type=begin
				##SBATCH --mail-type=end
				#SBATCH -A PADAS

				cd ~/projects/rsvd/build/
				ibrun ./rsvd_test --m $M --n $N --r $R --d $D --l $L --tol $TOL --adap ADAP --orient NORMAL --max_rank 500 --k 500 -q $Q

				exit 0
				EOS
			fi
			done
		done
	done
done
done
done
done
