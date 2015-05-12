#!/bin/bash
MAX_QUEUE=20
WAIT_TIME=120
MAX_TIME=04:00:00
TOL=0.00000001

round()
{
	echo $(printf %.$2f $(echo "scale=$2;(((10^$2)*$1)+0.5)/(10^$2)" | bc))
};

for NODES in 1 2
do
MD=$(echo "scale=3;sqrt($NODES)*10000" | bc)
M=$( round $MD 0)
ND=$(echo "scale=3;sqrt($NODES)*5000" | bc)
N=$( round $ND 0)
#	for M in $((10000*SCALE))
#	do
#		for N in $((5000*SCALE))
#		do
			for R in 25
			do
				for D in 0.85
				do
					for L in 10
					do
						while : ; do
							[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE ]] && break
							echo "Pausing until the queue empties enough to add a new one."
							sleep $WAIT_TIME
						done
						echo $N
						echo $M
						JOBNAME=weak8-exact-$NODES-$M-$N-$R-$D
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
						ibrun ./rsvd_test --m $M --n $N --r $R --d $D --l $L --tol $TOL --adap ADAP --orient NORMAL --max_rank 500 --k 500

						exit 0
						EOS
					done
				done
			done
		done
#	done
#done
