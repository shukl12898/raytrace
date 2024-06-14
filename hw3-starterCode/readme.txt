Assignment #3: Ray tracing

FULL NAME: Akanksha Shukla


MANDATORY FEATURES
------------------

<Under "Status" please indicate whether it has been implemented and is
functioning correctly.  If not, please explain the current status.>

Feature:                                 Status: finish? (yes/no)
-------------------------------------    -------------------------
1) Ray tracing triangles                  YES

2) Ray tracing sphere                     YES

3) Triangle Phong Shading                 YES

4) Sphere Phong Shading                   YES

5) Shadows rays                           YES

6) Still images                           YES
   
7) Extra Credit (up to 20 points)
All JPEG images with -extracredit include all three features. Else, I specify which features they include. The JPEG images without this are the core functionality of the program.
   - Recursive Reflection
      - I implemented recursive reflection with a maximum depth of 2 in order to balance reflected ray functionality and performance 
      - Created a new recursive_ray_trace() function, made modifications to Intersection struct, and added changes to intersect() function
   - Good Antialiasing
      - I used a 4x4 supersampling grid in order to generate multiple rays per pixel, each slightly offset to smooth the edges of the objects
      - I initially implemented this with an 8x8 supersampling grid, which looked much smoother, but after all the extra credits, it made performance too slow and I changed it to 2x2
      - All modifications in ray_trace() function 
   - Soft Shadows
      - I used a circular area light, and sampled random points within the area to create soft shadows instead of harsh shadows
      - Modified recursive_ray_trace(), shadow_test() and phong_shading() functions in order to implement
