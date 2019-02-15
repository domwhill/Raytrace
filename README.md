#--- 


#Ray trace Python test program 
Solves the ray trace equations to track the propagation and absorption of laser light through a plasma.
 This raytrace follows the algorithm outlined in: 
  Benkevitch et al. Algorithm for Tracing Radio Rays in Solar Corona
          .and. Chromosphere, arxiV:1006.5635v3 (2010) 



Currently set up to perform a reflection test:
Rays incident at different angles propagate up a linear density ramp. (from right to left)


#-------- Run instructions -----------
 Runs using python3 distribution
 Navigate to directory
 type in command line:
 ```sh 
  python raytrace_clean.py
 ```
 output: should generate a plot of ray positions, velocities and absorption within the computational domain.
 positions are compared with the analytic solution (black crosses).
